"""ASR: Whisper transcription + silero-vad loading.

Single transcribe() function with three filter layers:
1. RMS gate — skip Whisper entirely on near-silent input
2. Whisper confidence — skip on high no_speech_prob or low avg_logprob
3. Hallucination blacklist — last-resort substring match for known training pollution
"""

import numpy as np
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad


def load_whisper(model_name: str, device: str, compute_type: str) -> WhisperModel:
    print(f'Loading Whisper: {model_name} ({device}, {compute_type})...')
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print('Warming up Whisper...')
    model.transcribe(np.zeros(16000, dtype=np.float32))
    return model


def load_vad():
    print('Loading silero-vad (onnx)...')
    return load_silero_vad(onnx=True)


def strip_punct(s: str) -> str:
    """Remove spaces and common punctuation for fuzzy matching."""
    return ''.join(c for c in s if c not in ' ,.，。!?！？:;：；\'"\t\n')


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
    *,
    language: str | None = None,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    condition_on_previous_text: bool = False,
    temperature: float = 0.0,
    vad_filter: bool = False,
    rms_min: float = 0.005,
    no_speech_max: float = 1.0,
    avg_logprob_min: float = -10.0,
    hallucinations: list[str] = (),
) -> str:
    """Transcribe audio. Returns '' if filtered out by any layer.

    `hallucinations` should be a list of pre-stripped (via strip_punct) phrases.
    """
    # Layer 1: RMS gate
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    if rms < rms_min:
        return ''

    segs, _ = model.transcribe(
        audio,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        condition_on_previous_text=condition_on_previous_text,
        temperature=temperature,
        vad_filter=vad_filter,
    )
    segs = list(segs)
    if not segs:
        return ''

    # Layer 2: Whisper's own confidence signals
    no_speech = float(np.mean([s.no_speech_prob for s in segs]))
    avg_logp = float(np.mean([s.avg_logprob for s in segs]))
    if no_speech > no_speech_max:
        print(f'[low conf] no_speech_prob={no_speech:.2f} skipped')
        return ''
    if avg_logp < avg_logprob_min:
        print(f'[low conf] avg_logprob={avg_logp:.2f} skipped')
        return ''

    text = ''.join(s.text for s in segs).strip()

    # Layer 3: explicit blacklist (high-confidence training-data pollution)
    text_check = strip_punct(text)
    for h in hallucinations:
        if h and h in text_check:
            print(f'[blacklist] matched "{h}" skipped')
            return ''
    return text
