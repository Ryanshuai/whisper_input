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

import signal
import threading
import time

import numpy as np
import pyperclip
import sounddevice as sd
import yaml
from faster_whisper import WhisperModel
from pynput.keyboard import Controller, Key, KeyCode, Listener

# --- Config & Model ---

with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

model_name = cfg.get('model', 'base')
device = 'cpu' if cfg.get('compute_type') == 'int8' else cfg.get('device', 'auto')
compute = cfg.get('compute_type', 'default')
print(f'Loading model: {model_name} ({device}, {compute})...')
model = WhisperModel(model_name, device=device, compute_type=compute)
print('Warming up...')
model.transcribe(np.zeros(16000, dtype=np.float32))
print('Ready.')

# --- Screen Border Indicator ---

class ScreenBorder:
    """Shows a thin red border around all screens while recording."""

    def __init__(self, color='red', width=2):
        self._color = color
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
            bar.configure(bg=self._color)
            bar.attributes('-alpha', 0.8)
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

    def show(self):
        for bar in self._bars:
            bar.after(0, bar.deiconify)

    def hide(self):
        for bar in self._bars:
            bar.after(0, bar.withdraw)

# --- Recording & Transcription ---

kb = Controller()
recording = False
worker = None
border = ScreenBorder()

def run_pipeline():
    global recording
    recording = True
    border.show()
    print(f'\n{"-"*40}')
    print('Recording...')

    try:
        sr = cfg.get('sample_rate', 16000)
        chunks = []

        def cb(indata, *_):
            chunks.append(indata[:, 0].copy())

        with sd.InputStream(samplerate=sr, channels=1, dtype='int16', callback=cb):
            while recording:
                time.sleep(0.05)

        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
        dur = len(audio) / sr
        print(f'Recorded {dur:.1f}s')

        if dur < 0.3:
            print('Too short, skipped.')
            return

        print('Transcribing...')
        t0 = time.time()
        segs, _ = model.transcribe(
            audio.astype(np.float32) / 32768.0,
            language=cfg.get('language'),
            initial_prompt=cfg.get('initial_prompt'),
            condition_on_previous_text=cfg.get('condition_on_previous_text', True),
            temperature=cfg.get('temperature', 0.0),
            vad_filter=cfg.get('vad_filter', False),
        )
        text = ''.join(s.text for s in segs).strip()
        print(f'{time.time()-t0:.1f}s -> {text}')

        if text:
            pyperclip.copy(text)
            with kb.pressed(Key.ctrl):
                kb.tap(KeyCode.from_char('v'))
            time.sleep(0.05)
            kb.tap(Key.enter)
    except Exception as e:
        print(f'Error: {e}')
    finally:
        border.hide()
        recording = False

def on_hotkey():
    global recording, worker
    if recording:
        recording = False
        return
    if worker and worker.is_alive():
        return
    worker = threading.Thread(target=run_pipeline, daemon=True)
    worker.start()

# --- Hotkey Parsing & Main ---

def parse_hotkey(combo):
    key_str = combo.strip().lower()
    if hasattr(Key, key_str):
        return getattr(Key, key_str)
    if len(key_str) == 1:
        return KeyCode.from_char(key_str)
    raise ValueError(f'Unknown hotkey: {combo}')

if __name__ == '__main__':
    combo = cfg.get('activation_key', 'f13')
    hotkey = parse_hotkey(combo)
    print(f'Ready. [{combo}] to toggle recording.')
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    def on_press(key):
        if key == hotkey:
            on_hotkey()

    with Listener(on_press=on_press) as listener:
        listener.join()
