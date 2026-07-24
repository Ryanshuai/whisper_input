"""whisper-writer entry point.

Glue layer that wires together:
  asr.py         — pluggable ASR (faster_whisper | qwen3_asr)
  eou.py         — semantic end-of-utterance detection
  tts.py         — TTS playback + MCP tools exposed to Claude
  claude_chat.py — Claude SDK session

Owns the runtime state machine, audio loop, hotkey handlers, and dictation mode.
"""

import os
import sys

# On Linux, conda-forge's portaudio is ALSA-only and misses capture devices
# behind PipeWire/PulseAudio.  Override ctypes.util.find_library so that
# sounddevice loads the system libportaudio (which has JACK backend support).
# Combined with pw-jack, this lets portaudio see all PipeWire sources.
if sys.platform == 'linux':
    import glob as _glob, shutil as _shutil
    _sys_pa = next(iter(sorted(
        _glob.glob('/usr/lib/*/libportaudio.so*'))), None)
    if _sys_pa:
        import ctypes.util as _ctu
        _orig_find = _ctu.find_library
        _ctu.find_library = lambda name, _o=_orig_find: (
            _sys_pa if name == 'portaudio' else _o(name))
    if '_PW_JACK_DONE' not in os.environ:
        _pw_jack = _shutil.which('pw-jack')
        if _pw_jack:
            os.environ['_PW_JACK_DONE'] = '1'
            os.execv(_pw_jack, [_pw_jack, sys.executable] + sys.argv)

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

# Input-device selection is dynamic and self-healing: the audio supervisor
# (audio_loop) resolves a device on every (re)connect instead of pinning one at
# import time. That makes the mic robust to USB hot-plug and late enumeration on
# every platform sounddevice supports — Linux/PipeWire, Windows WASAPI/MME, macOS
# — because the reconnect trigger is a wall-clock "no-callback" watchdog, not an
# OS-specific hot-plug event.
_input_dev = cfg.get('input_device')

# Substrings of device names we never auto-select: phantom capture endpoints with
# no real mic behind them (HDMI inputs, internal HDA codecs). The configured
# `input_device` name and any USB-Audio mic still win over this list; it's only a
# last-resort guard so an empty config never grabs a dead card and records silence.
# Defaults are Linux/ALSA-flavored names; override via config `input_device_blocklist`.
_INPUT_BLOCKLIST = tuple(
    str(s).lower() for s in cfg.get('input_device_blocklist',
                                    ['hdmi', 'hda nvidia', 'hda intel pch'])
)
# Capture rates to try, SR (16k) first. Some mics reject 16k and only do 44.1/48k,
# in which case audio_loop captures at the native rate and resamples in the callback.
_SR_CANDIDATES = (SR, 48000, 44100, 32000, 22050)


def _list_input_devices():
    return [d for d in sd.query_devices() if d['max_input_channels'] > 0]


def _resolve_input_device():
    """Pick the best available capture device, or None if none is acceptable.

    Preference order: (1) the configured `input_device` entries — each a name
    substring or index; a YAML list is tried in order, so [DJI, Analog Stereo]
    means "DJI when plugged in, else the built-in mic"; (2) any USB-Audio mic;
    (3) any input not on the phantom blocklist. Returning None means 'wait for a
    real mic' — we never fall back to a phantom card that would record silence.
    """
    devs = _list_input_devices()
    if not devs:
        return None
    prefs = _input_dev if isinstance(_input_dev, (list, tuple)) else [_input_dev]
    for pref in prefs:
        if isinstance(pref, int):
            for d in devs:
                if d['index'] == pref:
                    return d
        elif pref:
            needle = str(pref).lower()
            for d in devs:
                if needle in d['name'].lower():
                    return d
    for d in devs:
        if 'usb' in d['name'].lower():
            return d
    for d in devs:
        if not any(b in d['name'].lower() for b in _INPUT_BLOCKLIST):
            return d
    return None


def _probe_hw_sr(dev_idx):
    """Lowest-friction capture rate the device accepts (SR preferred), or None if
    it supports none of the candidates."""
    for cand in _SR_CANDIDATES:
        try:
            sd.check_input_settings(device=dev_idx, channels=1,
                                    samplerate=cand, dtype='float32')
            return cand
        except Exception:
            continue
    return None
CLAUDE_ENABLED = bool(cfg.get('enable_claude', True))
HALLUCINATIONS = [asr.strip_punct(h) for h in cfg.get('hallucinations', []) if h.strip()]
WAKE_NAMES = [w.lower() for w in cfg.get('wake_names', [])]
WAKE_TRIGGERS_ON = [w.lower() for w in cfg.get('wake_triggers_on', [])]
WAKE_TRIGGERS_OFF = [w.lower() for w in cfg.get('wake_triggers_off', [])]
BACKCHANNEL = set(w.lower() for w in cfg.get('backchannel_words', []))
AUTO_OFF_SEC = cfg.get('auto_off_minutes', 5) * 60
LOG_PATH = os.path.join(os.path.dirname(__file__), cfg.get('log_file', 'conversations.jsonl'))


# ---------------- Models ----------------

asr_backend = asr.load_backend(cfg, HALLUCINATIONS)
vad_model = asr.load_vad()

if CLAUDE_ENABLED:
    print('Loading turn-detector (LiveKit EOU model, ~400MB, first run downloads)...')
    eou = EouDetector(
        threshold=float(cfg.get('eou_threshold', 0.2)),
        force_end_sec=float(cfg.get('eou_force_end_sec', 4.0)),
    )
    print(f'EOU ready (threshold={eou.threshold}, force_end={eou.force_end_sec}s)')
else:
    eou = None
    print('[Claude disabled] dictation-only mode — skipping EOU/TTS/Claude session.')


# ---------------- ScreenBorder UI ----------------

class ScreenBorder:
    """Thin border around all screens; color = current state.

    Thread-safe: show()/hide() only assign a desired-state field. All Tk calls
    run on the Tk thread via a self-scheduled poll loop, because Tkinter is not
    safe to call from other threads — cross-thread `.after()` can deadlock (which
    wedged stop_dictation before it reached transcription).
    """

    _MISSING = object()

    def __init__(self, width=3):
        self._width = width
        self._bars = []
        self._desired = None            # None = hidden; else color string
        self._applied = self._MISSING   # last state actually pushed to Tk
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
        self._poll()            # apply desired-state changes on the Tk thread
        self._tk.mainloop()

    def _poll(self):
        d = self._desired
        if d is not self._applied:
            self._applied = d
            for bar in self._bars:
                if d is None:
                    bar.withdraw()
                else:
                    bar.configure(bg=d)
                    bar.deiconify()
        self._tk.after(30, self._poll)

    def show(self, color):
        self._desired = color   # applied by _poll on the Tk thread

    def hide(self):
        self._desired = None


border = ScreenBorder()


# ---------------- TTS server + Claude session ----------------

if CLAUDE_ENABLED:
    _tts_server = tts.build_server(
        voices=cfg.get('tts_voices', {}),
        default_voice=cfg.get('tts_voice', 'Ryan'),
        starting_rate_pct=int(cfg.get('tts_rate_pct', 0)),
        starting_volume_pct=int(cfg.get('tts_volume_pct', 0)),
        backend=cfg.get('tts_backend', 'qwen3'),
        qwen_model_id=cfg.get('tts_qwen_model_id', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'),
        language_code=cfg.get('language', 'zh'),
    )

    # Read-only tools the assistant can use during voice chat. Web fetch + local
    # file inspection. Write/Edit/Bash deliberately omitted — voice chat doesn't
    # present a permission UI, so destructive ops shouldn't be reachable.
    _READ_TOOLS = ['WebSearch', 'WebFetch', 'Read', 'Grep', 'Glob']

    print('Starting Claude session...')
    session = claude_chat.ClaudeSession(
        system_prompt=claude_chat.SYSTEM_PROMPT,
        mcp_servers={'tts': _tts_server},
        allowed_tools=tts.TOOL_NAMES + _READ_TOOLS,
        permission_mode='bypassPermissions',
        model=cfg.get('chat_model'),  # None → CLI default; explicit name → that model
    )
    session.start()
    print('Claude SDK ready (uses local claude CLI OAuth, no API key needed).')
else:
    session = None


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


def _t() -> str:
    """HH:MM:SS stamp for dictation console lines, so they line up with the
    timestamps in dictation_log.tsv."""
    return datetime.now().strftime('%H:%M:%S')


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
    print(f'\n[{_t()}] >>> 语音输入模式 ON (recording)')


# --- Diagnostic (silent, no console output): log every dictation clip —
# timestamped wav + a TSV line (timestamp, wav name, out/raw/reason/conf) — so a
# hallucination reported as "a few minutes ago" can be located by time against
# the REAL mic audio instead of guessed at, and good cases promoted into the
# regression corpus. Persisted under `dictation_debug_dir` (config); set it empty
# to disable. Auto-prunes to the most recent _DICT_DUMP_KEEP clips. ---
_DICT_DUMP_DIR = cfg.get('dictation_debug_dir') or ''
_DICT_DUMP_KEEP = 300


def _dump_dictation(audio: np.ndarray, text: str, diag: dict | None = None):
    if not _DICT_DUMP_DIR:
        return
    try:
        import soundfile as sf
        os.makedirs(_DICT_DUMP_DIR, exist_ok=True)
        now = datetime.now()
        wav = f'{now.strftime("%Y%m%d-%H%M%S-%f")[:-3]}.wav'
        sf.write(os.path.join(_DICT_DUMP_DIR, wav), audio, SR)
        d = diag or {}
        # `out` is what was pasted (often '' when filtered); `raw` is Whisper's
        # unfiltered output — so a hallucination caught by the blacklist still has
        # its text + which layer killed it recorded here, not lost.
        raw = d.get('raw', text)
        reason = d.get('reason')
        with open(os.path.join(_DICT_DUMP_DIR, 'dictation_log.tsv'),
                  'a', encoding='utf-8') as f:
            f.write(f'{now.isoformat(timespec="milliseconds")}\t{wav}\t'
                    f'out={text!r}\traw={raw!r}\treason={reason}\t'
                    f'no_speech={d.get("no_speech")}\tavg_logprob={d.get("avg_logprob")}\t'
                    f'win_rms={d.get("win_rms")}\tvad={d.get("vad")}\n')
        wavs = sorted(p for p in os.listdir(_DICT_DUMP_DIR) if p.endswith('.wav'))
        for old in wavs[:-_DICT_DUMP_KEEP]:
            try:
                os.remove(os.path.join(_DICT_DUMP_DIR, old))
            except OSError:
                pass
    except Exception:
        pass


def stop_dictation():
    # test-and-set inside the lock so a concurrent caller (e.g. audio cb's
    # auto-stop on max-duration + user pressing Pause simultaneously) can't both
    # observe active=True and end up double-pasting.
    with Dict_.lock:
        if not Dict_.active:
            return
        Dict_.active = False
        chunks = Dict_.chunks[:]
        Dict_.chunks = []
    _audio_reset_request.set()  # flush VAD state so first post-dictation utterance is clean
    if not chunks:
        border.hide()
        print(f'[{_t()}] >>> 语音输入模式 OFF (no audio)')
        return
    audio = np.concatenate(chunks)
    duration = len(audio) / SR
    print(f'[{_t()}] >>> 语音输入模式 OFF, recorded {duration:.1f}s, transcribing...')
    if duration < 0.3:
        border.hide()
        print(f'[{_t()}]   too short, skipped')
        _dump_dictation(audio, '<too short>')
        return
    # Recording over; Whisper is now computing — switch border orange → blue until done.
    border.show('blue')
    try:
        diag: dict = {}
        text = _transcribe(audio, force=True, diag=diag)
    finally:
        border.hide()
    _dump_dictation(audio, text, diag)  # silent: stash real audio + raw result for diagnosis
    if not text:
        # "(empty)" alone is undebuggable. Use the diag to say WHY it was empty so
        # a dead mic is findable instead of mysterious.
        reason = diag.get('reason')
        win_rms = diag.get('win_rms') or 0.0
        if reason == 'rms_gate' and win_rms < 0.001:
            # Pure digital silence (win_rms≈0): the stream ran but every sample was
            # zero. A real mic always carries a noise floor, so this isn't a quiet
            # room — the source delivered nothing, almost always a wireless mic whose
            # transmitter is off / muted / out of range / flat.
            print(f'[{_t()}]   ⚠️  录到纯静音 (win_rms={win_rms:.4f}) — 麦克风没有真实音频输入!')
            print(f'[{_t()}]      多半是无线麦发射端掉线: 检查开机 / 电量 / 配对 / 是否误触静音键。')
        elif reason == 'rms_gate':
            print(f'[{_t()}]   ⚠️  音量太低 (win_rms={win_rms:.4f}) — 没说话? 或离麦太远 / 增益太小。')
        else:
            print(f'[{_t()}]   (empty, 被 {reason} 层拦下)')
        return
    print(f'[{_t()}] [Dictation] {text}')
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


# ---------------- Transcription wrapper ----------------

def _transcribe(audio: np.ndarray, force: bool = False, diag: dict | None = None) -> str:
    return asr_backend.transcribe(audio, force=force, diag=diag)


# ---------------- Speech segment handler (per silero-vad end event) ----------------

# Serializes all VAD-end handling. Without this two threads can each enter, both
# append to Pending.audio + transcribe the (now overlapping) buffer, and then
# one's EOU decision clears the buffer the other still depends on.
_handle_speech_lock = threading.Lock()


def handle_speech(audio: np.ndarray):
    if not CLAUDE_ENABLED:
        return
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

# No-callback-for-this-long ⇒ the device stopped delivering audio (USB unplugged,
# stream wedged). PortAudio fires the callback even during silence, so a gap this
# long is a real disconnect, not a quiet room — it triggers a reconnect. This is a
# plain wall-clock watchdog (not an OS hot-plug event), so it self-heals identically
# on Linux, Windows and macOS.
_AUDIO_STALL_SEC = 3.0

# Callbacks keep arriving but every sample is exactly zero for this long ⇒ the mic
# SOURCE is dead even though the USB device is alive. A wireless receiver whose
# transmitter dropped keeps streaming pure zeros; a real mic always carries a noise
# floor, so sustained exact-zero is a reliable "source lost" signal — distinct from
# _AUDIO_STALL_SEC (no callbacks at all = device unplugged). Without this the only
# symptom is silently-empty dictation, which is undebuggable from the user's side.
_SILENCE_WARN_SEC = 5.0


def audio_loop():
    """Supervised, self-healing capture loop. Resolves a mic, opens a stream, and
    runs the VAD pipeline until the device stalls or errors — then re-resolves and
    reconnects. If no acceptable mic exists it waits (polling) rather than grabbing
    a phantom card. Reconnect-on-loss only: a working mic is never pre-empted just
    because a more-preferred one was plugged in mid-session."""
    vad_iter = VADIterator(
        vad_model,
        sampling_rate=SR,
        threshold=cfg.get('vad_threshold', 0.5),
        min_silence_duration_ms=cfg.get('vad_min_silence_ms', 700),
    )
    chunk_samples = 512  # silero-vad expects 512 @ 16kHz
    PRE_ROLL_CHUNKS = 6  # ~192ms

    q: list = []
    q_lock = threading.Lock()
    waiting_logged = False

    while True:
        dev = _resolve_input_device()
        if dev is None:
            if not waiting_logged:
                print(f'[{_t()}] [Audio] 无可用麦克风,等待设备接入...')
                waiting_logged = True
            time.sleep(1.0)
            continue
        hw_sr = _probe_hw_sr(dev['index'])
        if not hw_sr:
            print(f"[{_t()}] [Audio] 设备 [{dev['index']}] {dev['name']} 不支持任何采样率,跳过")
            time.sleep(1.0)
            continue
        waiting_logged = False

        # Per-connection state, rebuilt on every (re)connect so a fresh stream
        # never inherits a stale resampler or partial chunk from a dead device.
        # If the mic can't capture at SR (16k) natively, capture at hw_sr and
        # streaming-resample to SR inside the callback. soxr.ResampleStream keeps
        # filter state across calls, so variable input block sizes are fine.
        need_resample = (hw_sr != SR)
        if need_resample:
            import soxr
            resampler = soxr.ResampleStream(hw_sr, SR, 1, dtype='float32', quality='HQ')
        else:
            resampler = None
        scratch = [np.empty(0, dtype=np.float32)]  # leftover 16k samples between callbacks
        last_cb = [time.monotonic()]
        last_nonzero = [time.monotonic()]  # last time a non-zero sample arrived (dead-source watchdog)
        silence_warned = [False]
        try:
            vad_iter.reset_states()
        except Exception:
            pass
        buffer: list = []
        in_speech = False
        pre_roll: list = []
        with q_lock:
            q.clear()

        def cb(indata, *_):
            last_cb[0] = time.monotonic()
            in_samples = indata[:, 0]
            out_samples = resampler.resample_chunk(in_samples) if need_resample else in_samples.copy()
            if len(out_samples) == 0:
                return
            if out_samples.any():  # any non-zero ⇒ source is live (a real mic's noise floor counts)
                last_nonzero[0] = time.monotonic()
            scratch[0] = np.concatenate([scratch[0], out_samples])
            while len(scratch[0]) >= chunk_samples:
                chunk = scratch[0][:chunk_samples].copy()
                scratch[0] = scratch[0][chunk_samples:]
                with q_lock:
                    q.append(chunk)
                if Dict_.active:
                    with Dict_.lock:
                        Dict_.chunks.append(chunk)
                        if time.time() - Dict_.started_at > DICT_MAX_SEC:
                            # Defer the actual stop_dictation work to a worker thread —
                            # don't run paste / Whisper inside the audio callback.
                            threading.Thread(target=stop_dictation, daemon=True).start()

        hw_blocksize = chunk_samples if not need_resample else int(round(chunk_samples * hw_sr / SR))
        try:
            with sd.InputStream(device=dev['index'], samplerate=hw_sr, channels=1,
                                dtype='float32', blocksize=hw_blocksize, callback=cb):
                host = sd.query_hostapis(dev['hostapi'])['name']
                tail = f' -> resampled to {SR}Hz mono' if need_resample else ''
                print(f"\n[{_t()}] [Audio] Listening on: [{dev['index']}] {dev['name']} "
                      f"({host}, native_sr={hw_sr}, in_ch={dev['max_input_channels']}){tail}")
                while True:
                    now = time.monotonic()
                    # Device-loss watchdog: no audio delivered ⇒ reconnect.
                    if now - last_cb[0] > _AUDIO_STALL_SEC:
                        print(f'[{_t()}] [Audio] 麦克风无数据(可能已拔出),重连中...')
                        break

                    # Dead-source watchdog: callbacks arriving but every sample is
                    # zero ⇒ the mic source is silent (wireless transmitter off / muted
                    # / out of range), not a quiet room. Warn once per silence episode;
                    # reset (and announce recovery) when real audio returns so a later
                    # drop warns again.
                    if now - last_nonzero[0] > _SILENCE_WARN_SEC:
                        if not silence_warned[0]:
                            silence_warned[0] = True
                            print(f'[{_t()}] [Audio] ⚠️  麦克风持续输出纯静音 '
                                  f'{_SILENCE_WARN_SEC:.0f}s+ — 发射端可能掉线 / 没电 / 被静音,'
                                  f'此时录音会全空。检查无线麦发射端。')
                    elif silence_warned[0]:
                        silence_warned[0] = False
                        print(f'[{_t()}] [Audio] ✅ 麦克风音频已恢复。')

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
        except Exception as e:
            # Stream open/close failed or the device vanished mid-stream. Re-resolve
            # and reconnect (or fall back to waiting) on the next loop iteration.
            print(f'[{_t()}] [Audio] 流错误: {e!r},重连中...')
        time.sleep(0.5)  # brief backoff so a hard-failing device can't busy-loop


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
    if CLAUDE_ENABLED:
        try:
            tts.stop_playback()
        except Exception:
            pass
    os._exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threading.Thread(target=audio_loop, daemon=True).start()
    if CLAUDE_ENABLED:
        threading.Thread(target=auto_off_loop, daemon=True).start()
        threading.Thread(target=turn_watcher_loop, daemon=True).start()

    combo_chat = cfg.get('activation_key') or '' if CLAUDE_ENABLED else ''
    combo_dict = cfg.get('dictation_key') or ''
    hotkey_chat = getattr(Key, combo_chat.lower()) if combo_chat else None
    hotkey_dict = getattr(Key, combo_dict.lower()) if combo_dict else None
    print('\nReady.')
    if hotkey_chat:
        print(f'  [{combo_chat}] toggle 对话模式 (Claude)')
    if CLAUDE_ENABLED:
        print('  对话模式语音切换：说 "Claude 开始" / "Claude 关闭"')
    if hotkey_dict:
        print(f'  [{combo_dict}] toggle 语音输入模式 (transcribe → paste)')

    def on_press(key):
        # pynput runs on_press inside the Windows low-level keyboard hook; if the
        # callback blocks past LowLevelHooksTimeout (~300ms) Windows silently drops
        # the hook and the hotkey dies. Transcription (Qwen3-ASR generate) takes
        # ~0.8s+, so dispatch handlers to worker threads to keep the hook instant.
        # start/stop_dictation already guard concurrency via Dict_.lock + test-and-set.
        if hotkey_chat and key == hotkey_chat:
            threading.Thread(target=on_chat_hotkey, daemon=True).start()
        elif hotkey_dict and key == hotkey_dict:
            threading.Thread(target=toggle_dictation, daemon=True).start()

    # Don't block the main thread in listener.join(): on Windows a C-level blocking
    # call swallows Ctrl+C (SIGINT is only delivered to the main thread between
    # bytecodes), so the app couldn't be exited. Poll in an interruptible sleep
    # loop instead — Ctrl+C then reaches _shutdown and exits cleanly.
    listener = Listener(on_press=on_press)
    listener.start()
    try:
        while listener.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
