import os
import signal
import threading
import time
from collections import deque

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
import webrtcvad
import yaml
from faster_whisper import WhisperModel

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))

print('Loading model...')
model = WhisperModel(
    cfg.get('model_path') or cfg.get('model', 'base'),
    device='cpu' if cfg.get('compute_type') == 'int8' else cfg.get('device', 'auto'),
    compute_type=cfg.get('compute_type', 'default'),
)
print('Model loaded.')

recording = False
worker = None

def run_pipeline():
    global recording
    recording = True
    print('Recording...')

    sr = cfg.get('sample_rate', 16000)
    frame_size = int(sr * 0.03)
    silence_limit = int(cfg.get('silence_duration', 900) / 30)
    skip = int(0.15 * sr / frame_size)
    vad = webrtcvad.Vad(2)
    speech_detected = False
    silent = 0
    buf = deque(maxlen=frame_size)
    frames = []
    ready = threading.Event()

    def cb(indata, *_):
        buf.extend(indata[:, 0])
        ready.set()

    with sd.InputStream(samplerate=sr, channels=1, dtype='int16', blocksize=frame_size, callback=cb):
        while recording:
            ready.wait(); ready.clear()
            if len(buf) < frame_size:
                continue
            frame = np.array(list(buf), dtype=np.int16)
            buf.clear()
            frames.extend(frame)
            if skip > 0:
                skip -= 1; continue
            if vad.is_speech(frame.tobytes(), sr):
                silent = 0
                if not speech_detected:
                    print('Speech detected.')
                    speech_detected = True
            else:
                silent += 1
            if speech_detected and silent > silence_limit:
                break

    audio = np.array(frames, dtype=np.int16)
    dur = len(audio) / sr
    print(f'Recorded {dur:.1f}s')
    if dur * 1000 < cfg.get('min_duration', 100):
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
    print(f'{time.time()-t0:.1f}s → {text}')
    if text:
        pyperclip.copy(text)
        keyboard.send('ctrl+v')

def on_hotkey():
    global recording, worker
    if recording:
        recording = False
        return
    if worker and worker.is_alive():
        return
    worker = threading.Thread(target=run_pipeline, daemon=True)
    worker.start()

if __name__ == '__main__':
    combo = cfg.get('activation_key', 'ctrl+alt+shift+t')
    keyboard.add_hotkey(combo, on_hotkey)
    print(f'Ready. [{combo}] to toggle recording.')
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    keyboard.wait()
