"""ASR: pluggable transcription backends + silero-vad loading.

Two backends, selected by config `backend`:
- faster_whisper (default): CTranslate2 Whisper. Torch-free. Keeps the original
  three-layer filtering and low-confidence gating.
- qwen3_asr: Qwen/Qwen3-ASR-0.6B-hf via transformers. Best at zh-primary speech
  with embedded English (code-switching). Lazily imports torch/transformers so
  the whisper path never pays for (or breaks on) the torch stack.

Shared filtering (both backends):
1. RMS gate      — skip transcription entirely on near-silent input
2. Hallucination blacklist — last-resort substring match for known training pollution
Whisper adds an extra confidence layer (no_speech_prob / avg_logprob) that has no
Qwen equivalent, so it lives inside the whisper backend.
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


class _Backend:
    """Shared energy gate + hallucination blacklist around a backend-specific decoder.

    `hallucinations` should be a list of pre-stripped (via strip_punct) phrases.

    Layer 1 (energy gate) and Layer 3 (blacklist) live here and run for every
    backend; the backend-specific `_decode` owns transcription and any Layer 2
    confidence gating it can compute (Whisper has one; Qwen has no equivalent).

    An optional `diag` dict is threaded through every layer so a caller can learn
    *why* the output was empty (dead mic vs. blacklist vs. low confidence) — see
    main.py's dictation diagnostics.
    """

    def __init__(self, *, rms_min: float, rms_peak_min: float, hallucinations: list[str]):
        self.rms_min = rms_min
        self.rms_peak_min = rms_peak_min
        self.hallucinations = hallucinations

    def transcribe(self, audio: np.ndarray, *, force: bool = False, diag: dict | None = None,
                   context: str | None = None) -> str:
        """Transcribe audio. Returns '' if filtered out by any layer.

        `context` is a free-text biasing hint describing what the user is
        currently working on (see asr_context.py). Each backend feeds it to its
        own biasing mechanism — Qwen's system prompt, Whisper's hotwords.

        When `force=True` (user-initiated dictation), the backend's averaged
        confidence checks (Layer 2) are skipped — pressing the hotkey signals real
        speech is expected, and whole-utterance averages can drop quiet-but-real
        dictation. The energy gate (Layer 1) and the blacklist (Layer 3) STILL
        run: silence should never paste, and known subtitle pollution
        ("感谢观看", "明镜与点点", ...) is never legitimate dictation, so the hotkey
        is no reason to paste (or Enter) it.
        """
        # Layer 1: energy gate (always on — even forced dictation shouldn't paste
        # silence). TWO complementary measures, because a single mean-RMS threshold
        # is length-dependent: a long clip padded with silence (or a brief quiet
        # word) gets its mean averaged below the floor even when real speech is
        # plainly there. That false-killed real dictation — e.g. a 27.6s clip whose
        # mean RMS was 0.00296 (< rms_min) yet carried a clear ~0.008 speech burst.
        # So we ALSO take the loudest ~200ms window (length-invariant) and skip ONLY
        # when BOTH the mean is low AND no window ever reached speech energy.
        # Empirically (captures corpus) low-energy noise hallucinations top out at
        # window-RMS ≈0.004 while the quietest real speech sits ≈0.008 — a ~2x gap
        # on each side of rms_peak_min, so this recovers diluted real speech without
        # letting the "请不吝点赞…" noise back in.
        audio = audio.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        win_rms = loudest_window_rms(audio)
        if diag is not None:
            diag['win_rms'] = win_rms
        if rms < self.rms_min and win_rms < self.rms_peak_min:
            if diag is not None:
                diag.update(raw='', reason='rms_gate', rms=rms)
            return ''

        text = self._decode(audio, force=force, rms=rms, diag=diag, context=context)
        if not text:
            return ''

        # Layer 3: explicit blacklist (known training-data pollution). ALWAYS on,
        # even under force — phrases like "感谢观看" / "明镜与点点" are never something
        # a user dictates, so the hotkey is no reason to paste (or Enter) them.
        text_check = strip_punct(text)
        for h in self.hallucinations:
            if h and h in text_check:
                print(f'[{time.strftime("%H:%M:%S")}] [blacklist] matched "{h}" skipped')
                if diag is not None:
                    diag['reason'] = f'blacklist:{h}'
                return ''
        return text

    def _decode(self, audio: np.ndarray, *, force: bool, rms: float, diag: dict | None,
                context: str | None) -> str:
        raise NotImplementedError


class FasterWhisperBackend(_Backend):
    def __init__(self, cfg: dict, hallucinations: list[str]):
        super().__init__(rms_min=cfg.get('rms_min', 0.005),
                         rms_peak_min=cfg.get('rms_peak_min', 0.005),
                         hallucinations=hallucinations)
        self.language = cfg.get('language')
        self.initial_prompt = cfg.get('initial_prompt')
        self.hotwords = cfg.get('hotwords')
        self.condition_on_previous_text = cfg.get('condition_on_previous_text', False)
        self.temperature = cfg.get('temperature', 0.0)
        self.vad_filter = cfg.get('vad_filter', False)
        self.vad_threshold = cfg.get('dictation_vad_threshold', 0.6)
        self.no_speech_max = cfg.get('no_speech_max', 1.0)
        self.avg_logprob_min = cfg.get('avg_logprob_min', -10.0)
        self.model = load_whisper(cfg['model'], cfg['device'], cfg['compute_type'])

    def _decode(self, audio: np.ndarray, *, force: bool, rms: float, diag: dict | None,
                context: str | None = None) -> str:
        # Dictation (force=True) has no external silero-vad segmentation, so the
        # blob carries leading/trailing silence + breaths that Whisper hallucinates
        # subtitle pollution on. Turn on faster-whisper's built-in VAD for that path
        # to strip non-speech before transcription. The streaming (Claude) path is
        # already tightly VAD-segmented upstream, so it keeps vad_filter as configured.
        use_vad = self.vad_filter or force

        # Whisper's only biasing hook is `hotwords` (a free string prepended in the
        # sot_prev slot), so the config hotwords and the live context share it.
        # faster-whisper truncates it to half the prompt window on its own.
        hotwords = ' '.join(p for p in (self.hotwords, context) if p) or None

        segs, _ = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=self.initial_prompt,
            hotwords=hotwords,
            condition_on_previous_text=self.condition_on_previous_text,
            temperature=self.temperature,
            vad_filter=use_vad,
            vad_parameters=dict(threshold=self.vad_threshold) if use_vad else None,
        )
        segs = list(segs)
        if not segs:
            if diag is not None:
                diag.update(raw='', reason='no_segments', rms=rms, vad=use_vad)
            return ''

        # Compute confidence + raw text up front so diag captures them even when a
        # later layer filters the output (e.g. a blacklisted hallucination still
        # gets its raw text recorded, not lost as an empty string).
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
            if no_speech > self.no_speech_max:
                print(f'[{time.strftime("%H:%M:%S")}] [low conf] no_speech_prob={no_speech:.2f} skipped')
                if diag is not None:
                    diag['reason'] = 'no_speech'
                return ''
            if avg_logp < self.avg_logprob_min:
                print(f'[{time.strftime("%H:%M:%S")}] [low conf] avg_logprob={avg_logp:.2f} skipped')
                if diag is not None:
                    diag['reason'] = 'avg_logprob'
                return ''

        return text


class Qwen3AsrBackend(_Backend):
    """Qwen3-ASR via transformers. torch/transformers imported lazily here so the
    faster_whisper path stays torch-free. Qwen has no per-utterance confidence
    signal (no_speech_prob / avg_logprob), so it relies only on the shared Layer 1
    energy gate and Layer 3 blacklist."""

    def __init__(self, cfg: dict, hallucinations: list[str]):
        super().__init__(rms_min=cfg.get('rms_min', 0.005),
                         rms_peak_min=cfg.get('rms_peak_min', 0.005),
                         hallucinations=hallucinations)
        self.model_id = cfg.get('qwen_asr_model', 'Qwen/Qwen3-ASR-0.6B-hf')
        # None → automatic language detection (best for zh+en code-switching);
        # or force e.g. "Chinese" / "English".
        self.language = cfg.get('qwen_asr_language') or None
        self.max_new_tokens = int(cfg.get('qwen_asr_max_new_tokens', 256))
        self.sample_rate = int(cfg.get('sample_rate', 16000))

        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        self._torch = torch
        device = cfg.get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            print('[Qwen3-ASR] CUDA unavailable, falling back to CPU (slow).')
            device = 'cpu'
        self.device = device
        self.dtype = torch.float16 if device == 'cuda' else torch.float32

        print(f'Loading Qwen3-ASR: {self.model_id} ({device}, {self.dtype})...')
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id, dtype=self.dtype
        ).to(device)
        self.model.eval()

        print('Warming up Qwen3-ASR...')
        self._decode(np.zeros(self.sample_rate, dtype=np.float32), force=True, rms=0.0, diag=None)

    def _decode(self, audio: np.ndarray, *, force: bool, rms: float, diag: dict | None,
                context: str | None = None) -> str:
        torch = self._torch
        # The chat template wants a bare 1-D array (already at the model's 16 kHz);
        # a (sr, ndarray) tuple is misread as a batch. We capture at 16 kHz.
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        # Contextual biasing lives in the **system** slot — that is Qwen3-ASR's
        # native mechanism, and the chat template just concatenates whatever text
        # is there. `apply_transcription_request` only ever puts the language name
        # in that slot, so with a context we build the conversation ourselves and
        # prepend the language (when forced) to keep both behaviours.
        system_text = '\n'.join(p for p in (self.language, context) if p)
        messages = []
        if system_text:
            messages.append({'role': 'system', 'content': [{'type': 'text', 'text': system_text}]})
        messages.append({'role': 'user', 'content': [{'type': 'audio', 'audio': audio}]})
        # BatchFeature.to(device, dtype) casts only float tensors, leaves ids intact.
        inputs = self.processor.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs['input_ids'].shape[1]:]
        text = self.processor.decode(gen, return_format='transcription_only')[0]
        text = text.strip()
        if diag is not None:
            # Qwen has no no_speech/avg_logprob; record what we can so the dictation
            # log/diagnostics stay uniform across backends.
            diag.update(raw=text, reason=None, rms=rms, vad=False,
                        no_speech=None, avg_logprob=None)
        return text


def load_backend(cfg: dict, hallucinations: list[str]) -> _Backend:
    backend = cfg.get('backend', 'faster_whisper')
    if backend == 'faster_whisper':
        return FasterWhisperBackend(cfg, hallucinations)
    if backend in ('qwen3_asr', 'qwen'):
        return Qwen3AsrBackend(cfg, hallucinations)
    raise ValueError(f'Unknown asr backend: {backend!r} (use faster_whisper | qwen3_asr)')
