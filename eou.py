"""LiveKit Turn Detector — semantic end-of-utterance detection.

Wraps the multilingual ONNX model from livekit/turn-detector. Used to decide
whether the user actually finished speaking after a brief silence, beyond what
silero-vad's pure-silence end detection can tell us.

Notes on the model:
- Repo: livekit/turn-detector
- Revision matters:
    * 'v0.4.1-intl' = multilingual, output shape (batch, seq_len) per-token probs
    * 'v1.2.2-en'   = English-only, output shape (1,) single prob
- LiveKit uses a CUSTOM chat template (NOT standard Qwen):
    <|im_start|><|user|>{content}<|im_end|>
- The trailing <|im_end|> must be stripped from the prompt — that's the token
  the model is predicting; including it forces the probability to ~0.
- Input text is NFKC-normalized + lowercased + most punctuation stripped
  (apostrophe and hyphen are kept).
"""

import os
import re
import unicodedata

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO_ID = 'livekit/turn-detector'
MODEL_FILE = 'model_q8.onnx'
DEFAULT_REVISION = 'v0.4.1-intl'

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


def _normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text.lower())
    text = ''.join(
        ch for ch in text
        if not (unicodedata.category(ch).startswith('P') and ch not in ("'", '-'))
    )
    return re.sub(r'\s+', ' ', text).strip()


class EouDetector:
    def __init__(self, threshold: float = 0.2, force_end_sec: float = 4.0,
                 revision: str = DEFAULT_REVISION):
        self.threshold = float(threshold)
        self.force_end_sec = float(force_end_sec)
        self.revision = revision
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            subfolder='onnx',
            revision=revision,
        )
        self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_pretrained(REPO_ID, revision=revision)

    def detect(self, text: str) -> float:
        """Return probability that user has finished speaking after `text`."""
        if not text.strip():
            return 1.0
        norm = _normalize(text)
        if not norm:
            return 1.0
        prompt = f'<|im_start|><|user|>{norm}'
        # Keep the TAIL (last 128 tokens), not the head — the model predicts
        # whether <|im_end|> follows the *latest* token, so the suffix is what
        # carries the signal. Truncating from the front loses everything useful.
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids[-128:]
        arr = np.array([ids], dtype=np.int64)
        out = self.session.run(None, {'input_ids': arr})
        # multilingual model output shape: (batch, seq_len) — take last position
        return float(out[0][0, -1])
