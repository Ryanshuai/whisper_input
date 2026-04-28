"""whisper-writer entry point.

Glue layer that wires together:
  asr.py         — Whisper transcription
  eou.py         — semantic end-of-utterance detection
  tts.py         — TTS playback + MCP tools exposed to Claude
  claude_chat.py — Claude SDK session

Owns the runtime state machine, audio loop, hotkey handlers, and dictation mode.
"""

import os
import sys

# Pre-load ctranslate2 DLLs on Windows (pixi env needs explicit DLL paths)
if sys.platform == 'win32':
    import ctypes
    import glob
    import importlib.resources
    os.add_dll_directory(os.path.join(os.environ['SystemRoot'], 'System32'))
    ct2_dir = str(importlib.resources.files('ctranslate2'))
    os.add_dll_directory(ct2_dir)
    for _dll in sorted(glob.glob(os.path.join(ct2_dir, '*.dll'))):
        ctypes.CDLL(_dll)

import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

# Load .env file if present so ANTHROPIC_API_KEY etc. don't need shell exports
_env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_file):
    with open(_env_file, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith('#') or '=' not in _line:
                continue
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import json
import re
import signal
import threading
import time
from datetime import datetime

import numpy as np
import pyperclip
import sounddevice as sd
import yaml
from pynput.keyboard import Controller as KbController, Key, KeyCode, Listener
from silero_vad import VADIterator

import asr
import claude_chat
import tts
from eou import EouDetector


# ---------------- Config ----------------

with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

SR = cfg.get('sample_rate', 16000)
HALLUCINATIONS = [asr.strip_punct(h) for h in cfg.get('hallucinations', []) if h.strip()]
WAKE_NAMES = [w.lower() for w in cfg.get('wake_names', [])]
WAKE_TRIGGERS_ON = [w.lower() for w in cfg.get('wake_triggers_on', [])]
WAKE_TRIGGERS_OFF = [w.lower() for w in cfg.get('wake_triggers_off', [])]
BACKCHANNEL = set(w.lower() for w in cfg.get('backchannel_words', []))
AUTO_OFF_SEC = cfg.get('auto_off_minutes', 5) * 60
LOG_PATH = os.path.join(os.path.dirname(__file__), cfg.get('log_file', 'conversations.jsonl'))


# ---------------- Models ----------------

whisper = asr.load_whisper(cfg['model'], cfg['device'], cfg['compute_type'])
vad_model = asr.load_vad()

print('Loading turn-detector (LiveKit EOU model, ~400MB, first run downloads)...')
eou = EouDetector(
    threshold=float(cfg.get('eou_threshold', 0.2)),
    force_end_sec=float(cfg.get('eou_force_end_sec', 4.0)),
)
print(f'EOU ready (threshold={eou.threshold}, force_end={eou.force_end_sec}s)')


# ---------------- ScreenBorder UI ----------------

class ScreenBorder:
    """Thin border around all screens; color = current state."""

    def __init__(self, width=3):
        self._width = width
        self._bars = []
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait()

    def _run(self):
        import tkinter as tk
        self._tk = tk.Tk()
        self._tk.withdraw()

        if sys.platform == 'win32':
            user32 = ctypes.windll.user32
            vx = user32.GetSystemMetrics(76)
            vy = user32.GetSystemMetrics(77)
            vw = user32.GetSystemMetrics(78)
            vh = user32.GetSystemMetrics(79)
        else:
            vx, vy = 0, 0
            vw = self._tk.winfo_screenwidth()
            vh = self._tk.winfo_screenheight()

        w = self._width
        for x, y, bw, bh in [
            (vx, vy, vw, w),
            (vx, vy + vh - w, vw, w),
            (vx, vy, w, vh),
            (vx + vw - w, vy, w, vh),
        ]:
            bar = tk.Toplevel(self._tk)
            bar.overrideredirect(True)
            bar.attributes('-topmost', True)
            bar.geometry(f'{bw}x{bh}+{x}+{y}')
            bar.attributes('-alpha', 0.85)
            if sys.platform == 'win32':
                bar.update_idletasks()
                hwnd = int(bar.wm_frame(), 16)
                style = user32.GetWindowLongW(hwnd, -20)
                user32.SetWindowLongW(hwnd, -20,
                                      style | 0x80000 | 0x20 | 0x8 | 0x80)
            bar.withdraw()
            self._bars.append(bar)

        self._ready.set()
        self._tk.mainloop()

    def show(self, color):
        for bar in self._bars:
            bar.after(0, lambda b=bar, c=color: (b.configure(bg=c), b.deiconify()))

    def hide(self):
        for bar in self._bars:
            bar.after(0, bar.withdraw)


border = ScreenBorder()


# ---------------- TTS server + Claude session ----------------

_tts_server = tts.build_server(
    voices=cfg.get('tts_voices', {}),
    default_voice=cfg.get('tts_voice', 'Ryan'),
    starting_rate_pct=int(cfg.get('tts_rate_pct', 0)),
    starting_volume_pct=int(cfg.get('tts_volume_pct', 0)),
    backend=cfg.get('tts_backend', 'qwen3'),
    qwen_model_id=cfg.get('tts_qwen_model_id', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'),
    language_code=cfg.get('language', 'zh'),
)

_NET_TOOLS = ['WebSearch', 'WebFetch']

print('Starting Claude session...')
session = claude_chat.ClaudeSession(
    system_prompt=claude_chat.SYSTEM_PROMPT,
    mcp_servers={'tts': _tts_server},
    allowed_tools=tts.TOOL_NAMES + _NET_TOOLS,
    permission_mode='bypassPermissions',
    model=cfg.get('chat_model'),  # None → CLI default; explicit name → that model
)
session.start()
print('Claude SDK ready (uses local claude CLI OAuth, no API key needed).')


# ---------------- Conversation state + log ----------------

class State:
    listen_mode: bool = False
    last_activity: float = 0.0
    claude_busy: bool = False


_log_lock = threading.Lock()
_session_id = datetime.now().strftime('%Y%m%d-%H%M%S')


def log_conversation(user_text, final_reply, intermediate, spoke_via_tool, err=None):
    entry = {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'session': _session_id,
        'user': user_text,
        'final_reply': final_reply,
        'spoke_via_tool': spoke_via_tool,
        'tts_voice': tts.TTSState.voice,
        'tts_rate_pct': tts.TTSState.rate_pct,
        'intermediate': intermediate,
    }
    if err:
        entry['error'] = err
    with _log_lock:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ---------------- Listen mode toggle ----------------

def set_listen_mode(on: bool, reason: str = '', greet: bool = True):
    """Toggle listen mode. greet=False suppresses the '我在'/'好的' confirmation
    (used when the wake utterance carried a real question we want to forward to
    Claude immediately — the greeting would step on the user's first turn)."""
    if State.listen_mode == on:
        return
    State.listen_mode = on
    State.last_activity = time.time()
    if on:
        border.show('green')
        print(f'\n>>> 对话模式 ON ({reason})')
        if greet:
            tts.speak_async('我在')
    else:
        border.hide()
        tts.stop_playback()
        print(f'\n>>> 对话模式 OFF ({reason})')
        threading.Thread(target=session.reset, daemon=True).start()
        if greet:
            tts.speak_async('好的')


# ---------------- Wake-word + backchannel matching ----------------

def _build_wake_regex(triggers: list) -> re.Pattern:
    """Match wake = name + trigger adjacent in original text, allowing only
    whitespace/punctuation between them. This rejects "我看了 Claude 文档要开始
    学习" (where "开始" is a separate clause from "Claude") while still matching
    "Claude 开始", "Claude，开始", "开始 Claude", "Hi Claude", etc.

    Boundary lookarounds reject substring matches like "Claudia 开始" (where
    'claude' is the prefix of 'claudia') or '开始一下' (where the trigger '开始'
    is part of a longer Chinese clause).
    """
    name_alts = '|'.join(re.escape(n.lower()) for n in WAKE_NAMES if n)
    trig_alts = '|'.join(re.escape(t.lower()) for t in triggers if t)
    if not name_alts or not trig_alts:
        return re.compile(r'(?!x)x')  # never matches
    sep = r'[\s,.，。!?！？:;：；]*'   # whitespace + common punctuation
    # English-only boundary on the OUTSIDE of the whole "name+sep+trig" group.
    # Per-token lookarounds would falsely reject "claude开始" (no whitespace,
    # name's last letter 'e' fails trigger's lookbehind). Only the outer edges
    # need to be non-letter to reject substring matches like "Claudia 开始".
    body = rf'(?:(?:{name_alts}){sep}(?:{trig_alts})|(?:{trig_alts}){sep}(?:{name_alts}))'
    return re.compile(rf'(?<![a-z]){body}(?![a-z])')


_RE_WAKE_ON = None
_RE_WAKE_OFF = None


def _wake_re_on():
    global _RE_WAKE_ON
    if _RE_WAKE_ON is None:
        _RE_WAKE_ON = _build_wake_regex(WAKE_TRIGGERS_ON)
    return _RE_WAKE_ON


def _wake_re_off():
    global _RE_WAKE_OFF
    if _RE_WAKE_OFF is None:
        _RE_WAKE_OFF = _build_wake_regex(WAKE_TRIGGERS_OFF)
    return _RE_WAKE_OFF


def match_wake_on(text: str) -> bool:
    return bool(_wake_re_on().search(text.lower()))


def match_wake_off(text: str) -> bool:
    return bool(_wake_re_off().search(text.lower()))


# ---------------- Chat orchestration ----------------

# Concurrency model: ClaudeSession.query holds its own lock, so concurrent chat()
# calls naturally serialize there. We don't need an extra _chat_lock at this layer.
# Barge-in uses session.cancel_inflight() to abort the in-flight query rather than
# rejecting the new turn — old turn raises QueryCancelled, new turn proceeds.

def chat(user_text: str):
    State.claude_busy = True
    State.last_activity = time.time()
    border.show('blue')
    tts.prewarm()  # wake Bluetooth audio during Claude's think time

    final_reply = ''
    intermediate = []
    spoke_via_tool = False
    err = None
    cancelled = False
    try:
        tts.SpeakState.spoke_via_tool = False
        final_reply, intermediate = session.query(user_text, timeout=60)
        final_reply = (final_reply or '').strip()
        spoke_via_tool = tts.SpeakState.spoke_via_tool

        if not final_reply and not spoke_via_tool:
            print('[Claude] (empty reply, nothing spoken)')
            return
        if final_reply:
            print(f'[Claude] {final_reply}')
        # If speak() tool was already invoked, don't auto-speak final_reply.
        if final_reply and not spoke_via_tool:
            tts.speak(final_reply)
    except claude_chat.QueryCancelled:
        cancelled = True
        err = 'cancelled (barge-in / reset)'
        print('[Claude] interrupted')
    except Exception as e:
        err = repr(e)
        print(f'[Claude error]: {e}')
    finally:
        try:
            log_conversation(user_text, final_reply, intermediate, spoke_via_tool, err)
        except Exception as e:
            print(f'[log warn]: {e}')
        State.claude_busy = False
        State.last_activity = time.time()
        # Cancellation could happen because listen_mode just toggled OFF; only
        # repaint border if we're still in conversation mode.
        if State.listen_mode and not cancelled:
            border.show('green')


# ---------------- Pending audio (EOU accumulation) ----------------

class Pending:
    audio: list = []
    last_end_ts: float = 0.0
    waiting: bool = False
    lock = threading.Lock()


def _clear_pending():
    with Pending.lock:
        Pending.audio = []
        Pending.waiting = False


# ---------------- Dictation mode (paste to active app) ----------------

class Dict_:
    active: bool = False
    chunks: list = []
    started_at: float = 0.0
    lock = threading.Lock()


DICT_MAX_SEC = float(cfg.get('dictation_max_seconds', 120))

# Set by start/stop_dictation; consumed by audio_loop to flush its VAD state and
# any in-flight speech buffer (otherwise the first utterance after dictation can
# get dropped or stitched together with stale chunks).
_audio_reset_request = threading.Event()


_kb = KbController()


def start_dictation():
    if Dict_.active:
        return
    if State.listen_mode:
        set_listen_mode(False, 'dictation start')
    with Dict_.lock:
        Dict_.chunks = []
        Dict_.active = True
        Dict_.started_at = time.time()
    _audio_reset_request.set()  # discard any partially-buffered VAD speech
    border.show('orange')
    print('\n>>> 语音输入模式 ON (recording)')


def stop_dictation():
    # test-and-set inside the lock so a concurrent caller (e.g. audio cb's
    # auto-stop on max-duration + user pressing F13 simultaneously) can't both
    # observe active=True and end up double-pasting.
    with Dict_.lock:
        if not Dict_.active:
            return
        Dict_.active = False
        chunks = Dict_.chunks[:]
        Dict_.chunks = []
    _audio_reset_request.set()  # flush VAD state so first post-dictation utterance is clean
    border.hide()
    if not chunks:
        print('>>> 语音输入模式 OFF (no audio)')
        return
    audio = np.concatenate(chunks)
    duration = len(audio) / SR
    print(f'>>> 语音输入模式 OFF, recorded {duration:.1f}s, transcribing...')
    if duration < 0.3:
        print('  too short, skipped')
        return
    text = _transcribe(audio)
    if not text:
        print('  (empty/hallucination)')
        return
    print(f'[Dictation] {text}')
    pyperclip.copy(text)
    with _kb.pressed(Key.ctrl):
        _kb.tap(KeyCode.from_char('v'))
    if cfg.get('dictation_press_enter', False):
        time.sleep(0.05)
        _kb.tap(Key.enter)


def toggle_dictation():
    if Dict_.active:
        stop_dictation()
    else:
        start_dictation()


# ---------------- Transcription wrapper (passes config to asr.transcribe) ----------------

def _transcribe(audio: np.ndarray) -> str:
    return asr.transcribe(
        whisper, audio,
        language=cfg.get('language'),
        initial_prompt=cfg.get('initial_prompt'),
        hotwords=cfg.get('hotwords'),
        condition_on_previous_text=cfg.get('condition_on_previous_text', False),
        temperature=cfg.get('temperature', 0.0),
        vad_filter=cfg.get('vad_filter', False),
        rms_min=cfg.get('rms_min', 0.005),
        no_speech_max=cfg.get('no_speech_max', 1.0),
        avg_logprob_min=cfg.get('avg_logprob_min', -10),
        hallucinations=HALLUCINATIONS,
    )


# ---------------- Speech segment handler (per silero-vad end event) ----------------

# Serializes all VAD-end handling. Without this two threads can each enter, both
# append to Pending.audio + transcribe the (now overlapping) buffer, and then
# one's EOU decision clears the buffer the other still depends on.
_handle_speech_lock = threading.Lock()


def handle_speech(audio: np.ndarray):
    if Dict_.active:
        # Dictation owns the mic; don't route VAD-segmented audio anywhere
        return

    duration_one = len(audio) / SR
    if duration_one < cfg.get('vad_min_speech_ms', 300) / 1000:
        return

    # Decision step — serialized. chat()/set_listen_mode are deferred until after
    # the lock is released so a long Claude call doesn't block VAD-end processing.
    text_to_chat = None
    listen_toggle = None  # None | True | False
    with _handle_speech_lock:
        with Pending.lock:
            Pending.audio.append(audio)
            full_audio = np.concatenate(Pending.audio)
            Pending.last_end_ts = time.time()

        full_dur = len(full_audio) / SR
        print(f'\n[Speech +{duration_one:.1f}s, total {full_dur:.1f}s] transcribing...')
        text = _transcribe(full_audio)
        if not text:
            _clear_pending()
            print('[empty/hallucination] cleared pending')
            return
        print(f'[Heard] {text}')

        if not State.listen_mode:
            if match_wake_on(text):
                _clear_pending()
                listen_toggle = True
                # If the user attached a real question to the wake phrase
                # (e.g. "Claude 开始查一下天气"), keep it as the first turn.
                if len(text.strip()) >= 11:
                    text_to_chat = text
            else:
                _clear_pending()
        else:
            State.last_activity = time.time()

            if match_wake_off(text):
                _clear_pending()
                listen_toggle = False
            elif text.strip().rstrip('。.,，!?！？').lower() in BACKCHANNEL:
                _clear_pending()
                print('[backchannel] ignored')
            else:
                eou_prob = eou.detect(text)
                if eou_prob >= eou.threshold:
                    _clear_pending()
                    print(f'[EOU done {eou_prob:.2f}] dispatching to Claude')
                    text_to_chat = text
                else:
                    with Pending.lock:
                        Pending.waiting = True
                    print(f'[EOU wait {eou_prob:.2f}] (continue listening up to {eou.force_end_sec}s)')

    # --- Outside the lock: side-effects that may take a long time ---
    if listen_toggle is True:
        set_listen_mode(True, 'wake word', greet=text_to_chat is None)
    elif listen_toggle is False:
        set_listen_mode(False, 'wake word')
    if text_to_chat:
        # Now that transcription has shown real intent (passed
        # wake/backchannel/EOU filters), commit to the switch: cancel any
        # in-flight query from a prior turn so the new one starts clean.
        # Deferred from VAD-start so noise/coughs don't kill live tasks.
        if State.claude_busy:
            print('[barge-in] cancelling prior query for new input')
            session.cancel_inflight()
        chat(text_to_chat)


def turn_watcher_loop():
    """Force-process pending audio if no new speech for force_end_sec."""
    while True:
        time.sleep(0.3)
        with Pending.lock:
            if not Pending.waiting:
                continue
            elapsed = time.time() - Pending.last_end_ts
            if elapsed < eou.force_end_sec:
                continue

        # Snapshot + clear pending under the heavy lock briefly, then release
        # before the (slow) Whisper transcribe so that handle_speech can keep
        # processing fresh VAD-end events for a new utterance.
        with _handle_speech_lock:
            with Pending.lock:
                if not Pending.waiting:
                    continue
                full_audio = (np.concatenate(Pending.audio)
                              if Pending.audio else np.array([], dtype=np.float32))
                Pending.audio = []
                Pending.waiting = False

        if len(full_audio) == 0:
            continue
        text = _transcribe(full_audio)
        if not text:
            continue
        print(f'[EOU forced after {elapsed:.1f}s silence] {text}')
        if State.listen_mode:
            if State.claude_busy:
                print('[barge-in] cancelling prior query for new input')
                session.cancel_inflight()
            chat(text)


# ---------------- Always-on audio loop ----------------

def audio_loop():
    vad_iter = VADIterator(
        vad_model,
        sampling_rate=SR,
        threshold=cfg.get('vad_threshold', 0.5),
        min_silence_duration_ms=cfg.get('vad_min_silence_ms', 700),
    )
    chunk_samples = 512  # silero-vad expects 512 @ 16kHz

    buffer: list = []
    in_speech = False
    pre_roll: list = []
    PRE_ROLL_CHUNKS = 6  # ~192ms

    q: list = []
    q_lock = threading.Lock()

    def cb(indata, *_):
        chunk = indata[:, 0].copy()
        with q_lock:
            q.append(chunk)
        if Dict_.active:
            with Dict_.lock:
                Dict_.chunks.append(chunk)
                if time.time() - Dict_.started_at > DICT_MAX_SEC:
                    # Defer the actual stop_dictation work to a worker thread —
                    # don't run paste / Whisper inside the audio callback.
                    threading.Thread(target=stop_dictation, daemon=True).start()

    with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                        blocksize=chunk_samples, callback=cb):
        print('\n[Audio] Listening on default input device.')
        while True:
            # Flush VAD pipeline if requested (dictation start/stop).
            if _audio_reset_request.is_set():
                _audio_reset_request.clear()
                buffer = []
                pre_roll = []
                in_speech = False
                try:
                    vad_iter.reset_states()
                except Exception:
                    pass
                with q_lock:
                    q.clear()

            with q_lock:
                chunk = q.pop(0) if q else None
            if chunk is None:
                time.sleep(0.005)
                continue
            if len(chunk) != chunk_samples:
                continue

            try:
                event = vad_iter(chunk, return_seconds=False)
            except Exception as e:
                print(f'[VAD error]: {e}')
                continue

            pre_roll.append(chunk)
            if len(pre_roll) > PRE_ROLL_CHUNKS:
                pre_roll.pop(0)

            if event and 'start' in event:
                in_speech = True
                buffer = list(pre_roll)
                # Barge-in: stop TTS immediately (cheap, user expects "speaking
                # → assistant shuts up"). Use non-blocking stop — blocking on
                # proc.wait() here would stall the mic queue and deform VAD
                # timing. Assumes headphones (otherwise speaker → mic loop).
                #
                # We do NOT cancel the in-flight Claude query yet — VAD start
                # can fire on noise / coughs / self-talk, and cancelling here
                # would kill a useful task before we know if this is real
                # input. The cancel decision is deferred to handle_speech,
                # after transcription, where we have actual text to inspect.
                if State.listen_mode:
                    tts.stop_playback_nowait()
            elif in_speech:
                buffer.append(chunk)

            if event and 'end' in event and in_speech:
                in_speech = False
                audio = np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)
                buffer = []
                threading.Thread(target=handle_speech, args=(audio,), daemon=True).start()


def auto_off_loop():
    while True:
        time.sleep(10)
        if State.listen_mode and not State.claude_busy:
            idle = time.time() - State.last_activity
            if idle > AUTO_OFF_SEC:
                set_listen_mode(False, f'auto-off after {AUTO_OFF_SEC // 60}min idle')


def on_chat_hotkey():
    set_listen_mode(not State.listen_mode, 'hotkey')


# ---------------- Main ----------------

def _shutdown(*_):
    try:
        tts.stop_playback()
    except Exception:
        pass
    os._exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threading.Thread(target=audio_loop, daemon=True).start()
    threading.Thread(target=auto_off_loop, daemon=True).start()
    threading.Thread(target=turn_watcher_loop, daemon=True).start()

    combo_chat = cfg.get('activation_key') or ''
    combo_dict = cfg.get('dictation_key') or ''
    hotkey_chat = getattr(Key, combo_chat.lower()) if combo_chat else None
    hotkey_dict = getattr(Key, combo_dict.lower()) if combo_dict else None
    print('\nReady.')
    if hotkey_chat:
        print(f'  [{combo_chat}] toggle 对话模式 (Claude)')
    print('  对话模式语音切换：说 "Claude 开始" / "Claude 关闭"')
    if hotkey_dict:
        print(f'  [{combo_dict}] toggle 语音输入模式 (transcribe → paste)')

    def on_press(key):
        if hotkey_chat and key == hotkey_chat:
            on_chat_hotkey()
        elif hotkey_dict and key == hotkey_dict:
            toggle_dictation()

    with Listener(on_press=on_press) as listener:
        listener.join()
