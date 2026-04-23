"""TTS: edge-tts synthesis + ffplay playback + MCP tools exposed to Claude.

This single module owns everything TTS-related:
- TTSState / SpeakState (mutable runtime state)
- speak / speak_async (playback API for the rest of the app)
- 8 MCP tools that let Claude control voice/rate/volume and trigger speak() itself
- build_server() factory called once during startup
"""

import asyncio
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


# --- Playback (edge-tts → ffplay subprocess) ---

_playback_proc: Optional[subprocess.Popen] = None
_playback_lock = threading.Lock()

# Silent MP3 used to pre-wake Bluetooth audio. A2DP power-saves after ~500ms of
# idle; first real TTS loses its head syllable during device wake. We fire this
# at chat() entry so by the time Claude finishes generating, the audio path is
# already warm. Real speak() replaces it via the normal stop_playback() flow.
_SILENCE_MP3: bytes = b''


async def _synthesize_mp3(text: str) -> bytes:
    rate = f'{TTSState.rate_pct:+d}%'
    volume = f'{TTSState.volume_pct:+d}%'
    audio = b''
    async for chunk in edge_tts.Communicate(
        text, TTSState.voice, rate=rate, volume=volume,
    ).stream():
        if chunk['type'] == 'audio':
            audio += chunk['data']
    return audio


def _play_mp3_blocking(mp3: bytes):
    """Play MP3 via ffplay, blocking until done or terminated."""
    global _playback_proc
    with _playback_lock:
        if _playback_proc and _playback_proc.poll() is None:
            _playback_proc.terminate()
            try:
                _playback_proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                _playback_proc.kill()
        _playback_proc = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', '-'],
            stdin=subprocess.PIPE,
        )
        proc = _playback_proc
    try:
        proc.stdin.write(mp3)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        return
    proc.wait()


def _play_mp3_if_idle(mp3: bytes):
    """Play MP3 only if nothing is currently playing. Used by prewarm — we
    must not stomp on real TTS if speak() happens to beat us to the lock."""
    global _playback_proc
    with _playback_lock:
        if _playback_proc and _playback_proc.poll() is None:
            return
        _playback_proc = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', '-'],
            stdin=subprocess.PIPE,
        )
        proc = _playback_proc
    try:
        proc.stdin.write(mp3)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        return
    proc.wait()


def prewarm():
    """Fire silent audio now so Bluetooth A2DP is awake when real TTS arrives.
    Call at chat() entry — by the time Claude finishes generating (500ms+),
    the audio path is up and the first syllable of the real reply is preserved.
    Real speak() replaces this via _play_mp3_blocking's terminate-old logic.
    """
    if not _SILENCE_MP3:
        return
    threading.Thread(target=_play_mp3_if_idle, args=(_SILENCE_MP3,), daemon=True).start()


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


def stop_playback():
    """Force-stop any in-flight TTS playback and wait for the worker to exit.
    Use from non-time-critical paths (e.g. before kicking off the next speak).
    """
    global _playback_proc
    with _playback_lock:
        proc = _playback_proc
        if not (proc and proc.poll() is None):
            return
        proc.terminate()
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass


def speak(text: str):
    """Synthesize and play `text`. Blocks calling thread until playback finishes."""
    if not text.strip():
        return
    try:
        # Use the persistent loop so edge-tts can re-use its aiohttp session;
        # avoids ~200-300ms loop-startup overhead on short replies like "好".
        fut = asyncio.run_coroutine_threadsafe(_synthesize_mp3(text), _tts_loop)
        mp3 = fut.result(timeout=15)
    except Exception as e:
        print(f'[TTS synth error]: {e}')
        return
    _play_mp3_blocking(mp3)


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
                 starting_rate_pct: int = 0, starting_volume_pct: int = 0):
    """Initialize TTS state and create MCP server. Call once during startup."""
    global _voices, _default_voice, _SILENCE_MP3
    _voices = dict(voices)
    _default_voice = default_voice

    TTSState.voice = default_voice
    TTSState.rate_pct = int(starting_rate_pct)
    TTSState.volume_pct = int(starting_volume_pct)

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
