"""ASR: Whisper transcription + silero-vad loading.

Single transcribe() function with three filter layers:
1. RMS gate — skip Whisper entirely on near-silent input
2. Whisper confidence — skip on high no_speech_prob or low avg_logprob
3. Hallucination blacklist — last-resort substring match for known training pollution
"""

import time

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


def loudest_window_rms(audio: np.ndarray, sr: int = 16000, win_sec: float = 0.2) -> float:
    """RMS of the loudest `win_sec` window — a length-invariant 'was there ever
    speech-level energy here?' measure.

    Whole-clip mean RMS is length-dependent: leading/trailing silence and long
    pauses dilute it, so a long-but-sparsely-voiced clip (or a short quiet word)
    averages below the gate even when real speech is plainly present. The loudest
    window doesn't get diluted — it asks only whether *any* ~200ms stretch reached
    speech energy. O(n) via a cumulative sum of squares; hop = win/2.
    """
    a = audio.astype(np.float32)
    w = int(win_sec * sr)
    if len(a) == 0:
        return 0.0
    if len(a) < w:
        return float(np.sqrt(np.mean(a ** 2)))
    cs = np.concatenate(([0.0], np.cumsum(a.astype(np.float64) ** 2)))
    starts = np.arange(0, len(a) - w + 1, max(1, w // 2))
    energies = (cs[starts + w] - cs[starts]) / w
    return float(np.sqrt(energies.max()))


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
    *,
    language: str | None = None,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    condition_on_previous_text: bool = False,
    temperature: float | list[float] = 0.0,
    vad_filter: bool = False,
    vad_threshold: float = 0.5,
    rms_min: float = 0.005,
    rms_peak_min: float = 0.005,
    no_speech_max: float = 1.0,
    avg_logprob_min: float = -10.0,
    hallucinations: list[str] = (),
    force: bool = False,
    diag: dict | None = None,
) -> str:
    """Transcribe audio. Returns '' if filtered out by any layer.

    `hallucinations` should be a list of pre-stripped (via strip_punct) phrases.

    When `force=True` (user-initiated dictation), the averaged Whisper
    confidence checks (Layer 2) are skipped — pressing the hotkey signals real
    speech is expected, and those whole-utterance averages can drop
    quiet-but-real dictation. But force also turns ON faster-whisper's built-in
    VAD to strip silence/breaths (the dictation blob has no external silero-vad
    segmentation), and the blacklist (Layer 3) STILL runs: known subtitle
    pollution ("感谢观看", "明镜与点点", ...) is never legitimate dictation, so the
    hotkey is no reason to paste it.
    """
    # Layer 1: energy gate (always on — even forced dictation shouldn't paste
    # silence). TWO complementary measures, because a single mean-RMS threshold is
    # length-dependent: a long clip padded with silence (or a brief quiet word)
    # gets its mean averaged below the floor even when real speech is plainly
    # there. That false-killed real dictation — e.g. a 27.6s clip whose mean RMS
    # was 0.00296 (< rms_min) yet carried a clear ~0.008 speech burst. So we ALSO
    # take the loudest ~200ms window (length-invariant) and skip ONLY when BOTH the
    # mean is low AND no window ever reached speech energy. Empirically (captures
    # corpus) low-energy noise hallucinations top out at window-RMS ≈0.004 while the
    # quietest real speech sits ≈0.008 — a ~2x gap on each side of rms_peak_min, so
    # this recovers diluted real speech without letting the "请不吝点赞…" noise back in.
    audio = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    win_rms = loudest_window_rms(audio)
    if diag is not None:
        diag['win_rms'] = win_rms
    if rms < rms_min and win_rms < rms_peak_min:
        if diag is not None:
            diag.update(raw='', reason='rms_gate', rms=rms)
        return ''

    # Dictation (force=True) has no external silero-vad segmentation, so the blob
    # carries leading/trailing silence + breaths that Whisper hallucinates
    # subtitle pollution on. Turn on faster-whisper's built-in VAD for that path
    # to strip non-speech before transcription. The streaming (Claude) path is
    # already tightly VAD-segmented upstream, so it keeps vad_filter as configured.
    use_vad = vad_filter or force

    segs, _ = model.transcribe(
        audio,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        condition_on_previous_text=condition_on_previous_text,
        temperature=temperature,
        vad_filter=use_vad,
        vad_parameters=dict(threshold=vad_threshold) if use_vad else None,
    )
    segs = list(segs)
    if not segs:
        if diag is not None:
            diag.update(raw='', reason='no_segments', rms=rms, vad=use_vad)
        return ''

    # Compute confidence + raw text up front so diag captures them even when a
    # later layer filters the output (e.g. a blacklisted hallucination still gets
    # its raw text recorded, not lost as an empty string).
    no_speech = float(np.mean([s.no_speech_prob for s in segs]))
    avg_logp = float(np.mean([s.avg_logprob for s in segs]))
    text = ''.join(s.text for s in segs).strip()
    if diag is not None:
        diag.update(raw=text, reason=None, rms=rms, vad=use_vad,
                    no_speech=no_speech, avg_logprob=avg_logp)

    if not force:
        # Layer 2: Whisper's own confidence signals. Skipped under force because
        # these average across the whole utterance and can drop quiet-but-real
        # dictation; the VAD strip above is the force path's source-side defense.
        if no_speech > no_speech_max:
            print(f'[{time.strftime("%H:%M:%S")}] [low conf] no_speech_prob={no_speech:.2f} skipped')
            if diag is not None:
                diag['reason'] = 'no_speech'
            return ''
        if avg_logp < avg_logprob_min:
            print(f'[{time.strftime("%H:%M:%S")}] [low conf] avg_logprob={avg_logp:.2f} skipped')
            if diag is not None:
                diag['reason'] = 'avg_logprob'
            return ''

    # Layer 3: explicit blacklist (known training-data pollution). ALWAYS on,
    # even under force — phrases like "感谢观看" / "明镜与点点" are never something a
    # user dictates, so pressing the hotkey is no reason to paste (or Enter) them.
    text_check = strip_punct(text)
    for h in hallucinations:
        if h and h in text_check:
            print(f'[{time.strftime("%H:%M:%S")}] [blacklist] matched "{h}" skipped')
            if diag is not None:
                diag['reason'] = f'blacklist:{h}'
            return ''
    return text
