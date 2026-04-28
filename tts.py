"""TTS: edge-tts OR local Qwen3-TTS-CustomVoice + ffplay playback + MCP tools.

This single module owns everything TTS-related:
- TTSState / SpeakState (mutable runtime state)
- speak / speak_async (playback API for the rest of the app)
- 8 MCP tools that let Claude control voice/rate/volume and trigger speak() itself
- build_server() factory called once during startup; picks backend ('edge' or 'qwen3')

Backend differences:
- edge-tts: free, reverse-engineered MS service. Native rate/volume via SSML
  percent strings. Returns MP3.
- qwen3: LOCAL inference of Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice on the same
  GPU as Whisper. ~6-8GB VRAM in bf16. No native rate/volume params, so we
  apply rate_pct/volume_pct via ffplay -af filters (atempo + volume). Output
  is 24kHz WAV bytes (encoded in-memory via soundfile). Model is loaded eagerly
  in build_server() to avoid 10s+ stall on first speak().
"""

import asyncio
import io
import os
import queue
import re
import subprocess
import threading
from typing import Optional

import edge_tts
from claude_agent_sdk import create_sdk_mcp_server, tool


# --- Background asyncio loop for edge-tts (avoids re-creating loop + WS each call) ---

_tts_loop = asyncio.new_event_loop()
threading.Thread(target=_tts_loop.run_forever, daemon=True).start()


# --- Mutable state ---

class TTSState:
    """TTS playback parameters; read by _synthesize_mp3 each call."""
    voice: str = 'zh-CN-XiaoxiaoNeural'
    rate_pct: int = 0
    volume_pct: int = 0


class SpeakState:
    """Tracks whether the speak() tool was invoked during the current chat round.
    main resets this to False before each chat() call, then reads it to decide
    whether to auto-speak the final TextBlock.
    """
    spoke_via_tool: bool = False


RATE_MIN, RATE_MAX = -50, 100
VOLUME_MIN, VOLUME_MAX = -50, 100

_voices: dict = {}
_default_voice: str = ''
_BACKEND: str = 'edge'   # 'edge' | 'qwen3'; set in build_server()


# --- Playback (edge-tts → ffplay subprocess) ---

_playback_proc: Optional[subprocess.Popen] = None
_playback_lock = threading.Lock()

# In-flight ws producer for the streaming edge path. We track it so
# stop_playback() can cancel the coroutine — otherwise barge-in would kill
# ffplay but the producer would keep consuming ws frames until its next
# stdin.write() fails, leaking a few hundred ms of work per cancel.
_streaming_task = None  # concurrent.futures.Future | None
_streaming_lock = threading.Lock()

# Silent MP3 used to pre-wake Bluetooth audio. A2DP power-saves after ~500ms of
# idle; first real TTS loses its head syllable during device wake. We fire this
# at chat() entry so by the time Claude finishes generating, the audio path is
# already warm. Real speak() replaces it via the normal stop_playback() flow.
_SILENCE_MP3: bytes = b''


async def _synthesize_edge(text: str) -> bytes:
    """edge-tts: native rate/volume via SSML percent strings."""
    rate = f'{TTSState.rate_pct:+d}%'
    volume = f'{TTSState.volume_pct:+d}%'
    audio = b''
    async for chunk in edge_tts.Communicate(
        text, TTSState.voice, rate=rate, volume=volume,
    ).stream():
        if chunk['type'] == 'audio':
            audio += chunk['data']
    return audio


# --- Local Qwen3-TTS model (loaded eagerly in build_server when backend=qwen3) ---

_qwen_model = None        # Qwen3TTSModel instance
_qwen_lang: str = 'Chinese'  # passed to generate_custom_voice()

# Map config['language'] (whisper-style ISO codes) → Qwen language string.
_QWEN_LANG_MAP = {
    'zh': 'Chinese', 'en': 'English', 'ja': 'Japanese', 'ko': 'Korean',
    'de': 'German', 'fr': 'French', 'ru': 'Russian', 'pt': 'Portuguese',
    'es': 'Spanish', 'it': 'Italian',
}


def _load_qwen_model(model_id: str, lang_code: str):
    """Eager-load Qwen3-TTS-CustomVoice. Called from build_server."""
    global _qwen_model, _qwen_lang
    print(f'Loading Qwen3-TTS ({model_id}) to GPU... (~3.5GB download first run, ~5-15s load)')
    import torch
    from qwen_tts import Qwen3TTSModel
    try:
        _qwen_model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map='cuda:0',
            dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        )
    except Exception as e:
        # FlashAttention2 install fails on Windows often; fall back to sdpa.
        print(f'[TTS] FlashAttention2 unavailable ({e.__class__.__name__}), falling back to sdpa')
        _qwen_model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map='cuda:0',
            dtype=torch.bfloat16,
            attn_implementation='sdpa',
        )
    _qwen_lang = _QWEN_LANG_MAP.get(lang_code, 'Chinese')
    print(f'Qwen3-TTS ready (language={_qwen_lang}, default voice={TTSState.voice})')


def _synthesize_qwen(text: str) -> bytes:
    """Local Qwen3-TTS-CustomVoice inference. Returns 24kHz WAV bytes.
    Rate/volume applied at playback time via ffplay -af, not here.
    """
    if _qwen_model is None:
        raise RuntimeError('Qwen3-TTS model not loaded; check build_server output for errors')
    import soundfile as sf
    wavs, sr = _qwen_model.generate_custom_voice(
        text=text,
        language=_qwen_lang,
        speaker=TTSState.voice,
    )
    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def _ffplay_audio_filter() -> Optional[str]:
    """For qwen backend, encode rate_pct/volume_pct as ffplay -af filter chain.
    Edge backend bakes them into the SSML, so returns None there.
    rate_pct -50..+100 → atempo 0.5..2.0; volume_pct same mapping.
    """
    if _BACKEND != 'qwen3':
        return None
    speed = (100 + TTSState.rate_pct) / 100.0
    volume = (100 + TTSState.volume_pct) / 100.0
    parts = []
    if abs(speed - 1.0) > 0.01:
        parts.append(f'atempo={speed:.2f}')
    if abs(volume - 1.0) > 0.01:
        parts.append(f'volume={volume:.2f}')
    return ','.join(parts) if parts else None


def _ffplay_cmd() -> list:
    cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet']
    af = _ffplay_audio_filter()
    if af:
        cmd.extend(['-af', af])
    cmd.append('-')
    return cmd


def _play_audio_blocking(audio: bytes):
    """Play audio bytes (mp3/wav, ffplay auto-detects), blocking until done."""
    global _playback_proc
    with _playback_lock:
        if _playback_proc and _playback_proc.poll() is None:
            _playback_proc.terminate()
            try:
                _playback_proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                _playback_proc.kill()
        _playback_proc = subprocess.Popen(_ffplay_cmd(), stdin=subprocess.PIPE)
        proc = _playback_proc
    try:
        proc.stdin.write(audio)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        return
    proc.wait()


def _play_audio_if_idle(audio: bytes):
    """Play only if nothing is currently playing. Used by prewarm — must not
    stomp on real TTS if speak() happens to beat us to the lock."""
    global _playback_proc
    with _playback_lock:
        if _playback_proc and _playback_proc.poll() is None:
            return
        # Prewarm uses silent MP3 with no filters needed, but go through the
        # same builder for consistency. atempo on silence is harmless.
        _playback_proc = subprocess.Popen(_ffplay_cmd(), stdin=subprocess.PIPE)
        proc = _playback_proc
    try:
        proc.stdin.write(audio)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        return
    proc.wait()


def prewarm():
    """Fire silent audio now so Bluetooth A2DP is awake when real TTS arrives.
    Call at chat() entry — by the time Claude finishes generating (500ms+),
    the audio path is up and the first syllable of the real reply is preserved.
    Real speak() replaces this via _play_audio_blocking's terminate-old logic.
    """
    if not _SILENCE_MP3:
        return
    threading.Thread(target=_play_audio_if_idle, args=(_SILENCE_MP3,), daemon=True).start()


def _cancel_streaming_task():
    """Cancel the in-flight edge-streaming producer coroutine, if any.
    Safe no-op when nothing is streaming. Without this, killing ffplay leaves
    the ws producer consuming frames until its next stdin.write() throws."""
    with _streaming_lock:
        task = _streaming_task
    if task and not task.done():
        task.cancel()


def stop_playback_nowait():
    """Fire-and-forget terminate. Use from latency-sensitive paths like the
    audio_loop's barge-in branch — blocking on proc.wait() there would stall
    the mic queue consumer for 500-700ms and deform VAD timing.
    """
    global _playback_proc
    with _playback_lock:
        proc = _playback_proc
        if proc and proc.poll() is None:
            proc.terminate()
    _cancel_streaming_task()


def stop_playback():
    """Force-stop any in-flight TTS playback and wait for the worker to exit.
    Use from non-time-critical paths (e.g. before kicking off the next speak).
    """
    global _playback_proc
    with _playback_lock:
        proc = _playback_proc
        if not (proc and proc.poll() is None):
            _cancel_streaming_task()  # producer may still run even if proc gone
            return
        proc.terminate()
    _cancel_streaming_task()
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass


def _play_edge_streaming_blocking(text: str) -> None:
    """edge-tts streaming playback: pipe ws audio chunks straight into ffplay
    stdin so playback starts before synthesis finishes. Blocks the calling
    thread until ffplay exits (natural end OR terminate from stop_playback).

    Why streaming matters: edge-tts.Communicate(...).stream() yields audio
    chunks as they arrive from the MS service. The previous batched impl
    awaited all chunks into a bytes buffer before any playback could start,
    so first-audio latency = full-synth latency (measured 700ms-3.7s by text
    length). Streaming brings first-audio down to ~one ws frame (300-500ms
    floor from connection setup, then incremental).

    Cancellation contract:
    - stop_playback*() terminates ffplay; the producer's next stdin.write
      raises BrokenPipeError and exits.
    - stop_playback*() also calls _cancel_streaming_task() so a producer
      currently awaiting ws.stream() (rather than writing) gets dropped
      immediately instead of leaking a frame's worth of work.
    """
    global _playback_proc, _streaming_task

    rate = f'{TTSState.rate_pct:+d}%'
    volume = f'{TTSState.volume_pct:+d}%'

    with _playback_lock:
        if _playback_proc and _playback_proc.poll() is None:
            _playback_proc.terminate()
            try:
                _playback_proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                _playback_proc.kill()
        _playback_proc = subprocess.Popen(_ffplay_cmd(), stdin=subprocess.PIPE)
        proc = _playback_proc

    async def producer():
        try:
            async for chunk in edge_tts.Communicate(
                text, TTSState.voice, rate=rate, volume=volume,
            ).stream():
                if chunk['type'] != 'audio':
                    continue
                try:
                    proc.stdin.write(chunk['data'])
                except (BrokenPipeError, OSError):
                    return  # ffplay was killed (barge-in / next speak)
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f'[TTS edge stream error]: {e}')

    fut = asyncio.run_coroutine_threadsafe(producer(), _tts_loop)
    with _streaming_lock:
        _streaming_task = fut
    try:
        proc.wait()
    finally:
        # Drop the producer if it's still mid-await (e.g. ws hasn't sent the
        # next frame yet). Cancel is harmless when already finished.
        if not fut.done():
            fut.cancel()
        try:
            fut.result(timeout=1)
        except Exception:
            pass
        with _streaming_lock:
            if _streaming_task is fut:
                _streaming_task = None


def _synth_one(text: str) -> bytes:
    """Synthesize a single chunk with the active backend."""
    if _BACKEND == 'qwen3':
        return _synthesize_qwen(text)
    fut = asyncio.run_coroutine_threadsafe(_synthesize_edge(text), _tts_loop)
    return fut.result(timeout=15)


# Sentence boundary: split AFTER terminal punctuation (zh + en).
# Comma/semicolon also break — long Claude replies often have one giant comma-
# separated clause; chunking on commas gets first-audio out faster.
_SENT_SPLIT = re.compile(r'(?<=[。！？!?；;，,])\s*')

# Min chunk length: avoid synthesizing single-char fragments (overhead per call
# on Qwen3-TTS is ~500ms regardless of text length).
_MIN_CHUNK_CHARS = 8

# Only stream when total text exceeds this — short replies (e.g. "我在", "好的")
# pay overhead with no perceived-latency benefit from chunking.
_STREAM_MIN_CHARS = 25


def _split_for_streaming(text: str) -> list:
    """Split text into chunks for streamed synthesis. Merges fragments shorter
    than _MIN_CHUNK_CHARS into the next chunk so we don't fire tiny synth calls.
    """
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not parts:
        return [text]
    merged = []
    buf = ''
    for p in parts:
        buf = (buf + p) if buf else p
        if len(buf) >= _MIN_CHUNK_CHARS:
            merged.append(buf)
            buf = ''
    if buf:
        if merged:
            merged[-1] = merged[-1] + buf  # tail too short → glue to last
        else:
            merged.append(buf)
    return merged


def speak(text: str):
    """Synthesize and play `text`. Blocks calling thread until playback finishes.

    For long text on the qwen3 backend (slow local inference), streams chunks:
    splits on punctuation, synthesizes sentence-by-sentence, and starts playing
    the first chunk while later chunks are still being generated. Cuts perceived
    first-audio latency from "wait for full synth" to "wait for first sentence".
    """
    if not text.strip():
        return

    # edge-tts: chunk-level streaming for all lengths. ws audio frames are
    # piped straight into ffplay stdin, so first-audio latency drops to one
    # frame (~300-500ms ws floor) instead of full-synth latency.
    if _BACKEND == 'edge':
        _play_edge_streaming_blocking(text)
        return

    # qwen3 (local inference): single-shot for short text — chunking < 25 chars
    # adds per-call overhead with no perceived-latency win.
    if len(text) < _STREAM_MIN_CHARS:
        try:
            audio = _synth_one(text)
        except Exception as e:
            print(f'[TTS synth error]: {e}')
            return
        _play_audio_blocking(audio)
        return

    # qwen3 streamed: producer thread synthesizes chunks → main thread plays.
    chunks = _split_for_streaming(text)
    if len(chunks) == 1:
        try:
            audio = _synth_one(chunks[0])
        except Exception as e:
            print(f'[TTS synth error]: {e}')
            return
        _play_audio_blocking(audio)
        return

    audio_q: queue.Queue = queue.Queue(maxsize=4)
    _SENTINEL = object()

    def producer():
        for ch in chunks:
            try:
                audio = _synth_one(ch)
            except Exception as e:
                print(f'[TTS synth chunk error]: {e}')
                break
            audio_q.put(audio)
        audio_q.put(_SENTINEL)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        item = audio_q.get()
        if item is _SENTINEL:
            return
        _play_audio_blocking(item)


def speak_async(text: str):
    """Non-blocking speak: kicks off a daemon thread.

    Kills any in-flight playback first so two consecutive speak_async calls
    (e.g. '我在' + Claude's first reply) don't both play at the same time —
    without this, _playback_proc gets overwritten and the older proc keeps
    playing while stop_playback() only targets the newer one.
    """
    stop_playback()
    threading.Thread(target=speak, args=(text,), daemon=True).start()


# --- MCP tools (exposed to Claude) ---

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _ok(text: str) -> dict:
    return {'content': [{'type': 'text', 'text': text}]}


@tool(
    'get_tts_settings',
    '查询当前 TTS 语音输出的状态：当前音色、当前语速百分比、当前音量百分比，以及所有可用的音色标签列表。'
    '当用户问"有几种音色""可以选什么声音""现在用什么声音"等时调用。',
    {},
)
async def _t_get_tts_settings(args):
    voices_lines = '\n'.join(f'- {label}: {vid}' for label, vid in _voices.items())
    return _ok(
        f'当前音色: {TTSState.voice}\n'
        f'当前语速: {TTSState.rate_pct:+d}% (范围 {RATE_MIN} 到 {RATE_MAX})\n'
        f'当前音量: {TTSState.volume_pct:+d}% (范围 {VOLUME_MIN} 到 {VOLUME_MAX})\n'
        f'可用音色 ({len(_voices)} 种):\n{voices_lines}'
    )


@tool(
    'set_voice',
    '切换 TTS 音色。label 是音色标签（英文），必须是 get_tts_settings 返回的可用音色之一。',
    {'label': str},
)
async def _t_set_voice(args):
    label = str(args.get('label', '')).strip()
    if label not in _voices:
        return _ok(f'未知音色 "{label}"。可用: {list(_voices.keys())}')
    TTSState.voice = _voices[label]
    return _ok(f'已切换音色 → {label} ({TTSState.voice})')


@tool(
    'adjust_rate',
    f'相对调整语速。delta 是百分比变化量，正数加快，负数减慢。范围 {RATE_MIN} 到 {RATE_MAX}。',
    {'delta': int},
)
async def _t_adjust_rate(args):
    delta = int(args.get('delta', 0))
    TTSState.rate_pct = _clamp(TTSState.rate_pct + delta, RATE_MIN, RATE_MAX)
    return _ok(f'语速调整后: {TTSState.rate_pct:+d}%')


@tool(
    'set_rate',
    f'直接设定语速百分比。value 范围 {RATE_MIN} 到 {RATE_MAX}，0 是默认。',
    {'value': int},
)
async def _t_set_rate(args):
    val = _clamp(int(args.get('value', 0)), RATE_MIN, RATE_MAX)
    TTSState.rate_pct = val
    return _ok(f'语速已设为: {TTSState.rate_pct:+d}%')


@tool(
    'adjust_volume',
    f'相对调整音量。delta 是百分比变化量。范围 {VOLUME_MIN} 到 {VOLUME_MAX}。',
    {'delta': int},
)
async def _t_adjust_volume(args):
    delta = int(args.get('delta', 0))
    TTSState.volume_pct = _clamp(TTSState.volume_pct + delta, VOLUME_MIN, VOLUME_MAX)
    return _ok(f'音量调整后: {TTSState.volume_pct:+d}%')


@tool(
    'set_volume',
    f'直接设定音量百分比。value 范围 {VOLUME_MIN} 到 {VOLUME_MAX}。',
    {'value': int},
)
async def _t_set_volume(args):
    val = _clamp(int(args.get('value', 0)), VOLUME_MIN, VOLUME_MAX)
    TTSState.volume_pct = val
    return _ok(f'音量已设为: {TTSState.volume_pct:+d}%')


@tool('reset_tts', '重置 TTS 设置为默认（语速 0%、音量 0%、起始音色）。', {})
async def _t_reset_tts(args):
    TTSState.voice = _default_voice
    TTSState.rate_pct = 0
    TTSState.volume_pct = 0
    return _ok('已重置')


@tool(
    'speak',
    '触发 TTS 朗读传入的文本（精简口播版本）。一旦调用此工具，你最终的 TextBlock 内容将不再被自动朗读，'
    '只会在终端显示。用法：当你的回答包含详细信息（搜索结果、数据、来源）但用户只需要听到简短结论时，'
    '把详细内容写在普通 TextBlock 里（仅显示），用 speak("简短的1-2句口播") 单独触发朗读。',
    {'text': str},
)
async def _t_speak(args):
    text = str(args.get('text', '')).strip()
    if not text:
        return _ok('（空字符串，未朗读）')
    speak_async(text)
    SpeakState.spoke_via_tool = True
    return _ok(f'已朗读: {text}')


TOOL_NAMES = [
    'mcp__tts__get_tts_settings',
    'mcp__tts__set_voice',
    'mcp__tts__adjust_rate',
    'mcp__tts__set_rate',
    'mcp__tts__adjust_volume',
    'mcp__tts__set_volume',
    'mcp__tts__reset_tts',
    'mcp__tts__speak',
]


def build_server(voices: dict, default_voice: str,
                 starting_rate_pct: int = 0, starting_volume_pct: int = 0,
                 backend: str = 'qwen3',
                 qwen_model_id: str = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
                 language_code: str = 'zh'):
    """Initialize TTS state and create MCP server. Call once during startup."""
    global _voices, _default_voice, _SILENCE_MP3, _BACKEND
    _voices = dict(voices)
    _default_voice = default_voice
    _BACKEND = backend if backend in ('edge', 'qwen3') else 'edge'

    TTSState.voice = default_voice
    TTSState.rate_pct = int(starting_rate_pct)
    TTSState.volume_pct = int(starting_volume_pct)

    if _BACKEND == 'qwen3':
        try:
            _load_qwen_model(qwen_model_id, language_code)
        except Exception as e:
            print(f'[TTS] FATAL: Qwen3-TTS load failed ({e}); speak() will error. '
                  f'To fall back to edge-tts: set tts_backend: edge in config.yaml')

    try:
        _SILENCE_MP3 = subprocess.run(
            ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
             '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono',
             '-t', '60', '-acodec', 'libmp3lame', '-b:a', '32k', '-f', 'mp3', '-'],
            capture_output=True, check=True, timeout=10,
        ).stdout
    except Exception as e:
        print(f'[TTS prewarm setup warn]: {e} (BT first-syllable clipping may occur)')

    return create_sdk_mcp_server(
        name='tts',
        version='1.0.0',
        tools=[
            _t_get_tts_settings, _t_set_voice,
            _t_adjust_rate, _t_set_rate,
            _t_adjust_volume, _t_set_volume,
            _t_reset_tts,
            _t_speak,
        ],
    )
