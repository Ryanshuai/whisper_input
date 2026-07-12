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


class _Backend:
    """Shared RMS gate + hallucination blacklist around a backend-specific decoder.

    `hallucinations` should be a list of pre-stripped (via strip_punct) phrases.
    """

    def __init__(self, *, rms_min: float, hallucinations: list[str]):
        self.rms_min = rms_min
        self.hallucinations = hallucinations

    def transcribe(self, audio: np.ndarray, *, force: bool = False) -> str:
        """Transcribe audio. Returns '' if filtered out by any layer.

        When `force=True`, only the RMS silence gate is applied; the backend's
        confidence checks and the hallucination blacklist are skipped. Use this
        for user-initiated dictation where pressing the hotkey is itself a signal
        that real speech is expected and false-positive filtering does more harm
        than good.
        """
        # Layer 1: RMS gate (always on — even forced dictation shouldn't paste silence)
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        if rms < self.rms_min:
            return ''

        text = self._decode(audio, force=force)
        if not text:
            return ''

        if not force:
            # Layer 3: explicit blacklist (high-confidence training-data pollution)
            text_check = strip_punct(text)
            for h in self.hallucinations:
                if h and h in text_check:
                    print(f'[blacklist] matched "{h}" skipped')
                    return ''
        return text

    def _decode(self, audio: np.ndarray, *, force: bool) -> str:
        raise NotImplementedError


class FasterWhisperBackend(_Backend):
    def __init__(self, cfg: dict, hallucinations: list[str]):
        super().__init__(rms_min=cfg.get('rms_min', 0.005), hallucinations=hallucinations)
        self.language = cfg.get('language')
        self.initial_prompt = cfg.get('initial_prompt')
        self.hotwords = cfg.get('hotwords')
        self.condition_on_previous_text = cfg.get('condition_on_previous_text', False)
        self.temperature = cfg.get('temperature', 0.0)
        self.vad_filter = cfg.get('vad_filter', False)
        self.no_speech_max = cfg.get('no_speech_max', 1.0)
        self.avg_logprob_min = cfg.get('avg_logprob_min', -10.0)
        self.model = load_whisper(cfg['model'], cfg['device'], cfg['compute_type'])

    def _decode(self, audio: np.ndarray, *, force: bool) -> str:
        segs, _ = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=self.initial_prompt,
            hotwords=self.hotwords,
            condition_on_previous_text=self.condition_on_previous_text,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
        )
        segs = list(segs)
        if not segs:
            return ''

        if not force:
            # Layer 2: Whisper's own confidence signals
            no_speech = float(np.mean([s.no_speech_prob for s in segs]))
            avg_logp = float(np.mean([s.avg_logprob for s in segs]))
            if no_speech > self.no_speech_max:
                print(f'[low conf] no_speech_prob={no_speech:.2f} skipped')
                return ''
            if avg_logp < self.avg_logprob_min:
                print(f'[low conf] avg_logprob={avg_logp:.2f} skipped')
                return ''

        return ''.join(s.text for s in segs).strip()


class Qwen3AsrBackend(_Backend):
    """Qwen3-ASR via transformers. torch/transformers imported lazily here so the
    faster_whisper path stays torch-free."""

    def __init__(self, cfg: dict, hallucinations: list[str]):
        super().__init__(rms_min=cfg.get('rms_min', 0.005), hallucinations=hallucinations)
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
        self._decode(np.zeros(self.sample_rate, dtype=np.float32), force=True)

    def _decode(self, audio: np.ndarray, *, force: bool) -> str:
        torch = self._torch
        # apply_transcription_request wants a bare 1-D array (already at the model's
        # 16 kHz); a (sr, ndarray) tuple is misread as a batch. We capture at 16 kHz.
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        req = {'audio': audio}
        if self.language:
            req['language'] = self.language
        # BatchFeature.to(device, dtype) casts only float tensors, leaves ids intact.
        inputs = self.processor.apply_transcription_request(**req).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs['input_ids'].shape[1]:]
        text = self.processor.decode(gen, return_format='transcription_only')[0]
        return text.strip()


def load_backend(cfg: dict, hallucinations: list[str]) -> _Backend:
    backend = cfg.get('backend', 'faster_whisper')
    if backend == 'faster_whisper':
        return FasterWhisperBackend(cfg, hallucinations)
    if backend in ('qwen3_asr', 'qwen'):
        return Qwen3AsrBackend(cfg, hallucinations)
    raise ValueError(f'Unknown asr backend: {backend!r} (use faster_whisper | qwen3_asr)')
