"""Streaming speech segmentation with the Silero VAD model shipped inside faster-whisper."""
import os

import numpy as np
import onnxruntime
from faster_whisper.utils import get_assets_path

FRAME = 512  # samples @16 kHz = 32 ms
CONTEXT = 64
FRAME_MS = FRAME / 16000 * 1000


class _StreamingSilero:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), "silero_vad_v6.onnx"),
            providers=["CPUExecutionProvider"], sess_options=opts,
        )
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.ctx = np.zeros(CONTEXT, dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.ctx, frame])[None, :].astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.ctx = frame[-CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


class AutoGain:
    """Boost quiet audio toward a fixed level so soft speech still trips the VAD.
    Fast attack / slow release peak follower; near-silence is left alone so noise isn't blown up."""

    def __init__(self, target_peak=0.5, max_gain=30.0, floor=3e-4, release_s=3.0):
        self.target = target_peak
        self.max_gain = max_gain
        self.floor = floor
        self.decay = 0.5 ** (FRAME / 16000 / release_s)
        self.env = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(frame))) if len(frame) else 0.0
        self.env = max(peak, self.env * self.decay)
        if self.env < self.floor:
            return frame
        gain = min(self.max_gain, max(1.0, self.target / self.env))
        return np.clip(frame * gain, -1.0, 1.0)


class SpeechDetector:
    """Feed 512-sample frames; returns a finished utterance (np.ndarray) when one ends."""

    def __init__(self, threshold=0.5, silence_ms=500, max_segment_s=8.0, preroll_ms=300, min_speech_ms=250):
        self.model = _StreamingSilero()
        self.on = threshold
        self.off = max(0.15, threshold - 0.15)
        self.silence_frames = int(silence_ms / FRAME_MS)
        self.max_frames = int(max_segment_s * 1000 / FRAME_MS)
        self.preroll_frames = int(preroll_ms / FRAME_MS)
        self.min_frames = int(min_speech_ms / FRAME_MS)
        self.preroll: list[np.ndarray] = []
        self.buf: list[np.ndarray] = []
        self.probs: list[float] = []
        self.in_speech = False
        self.quiet = 0
        self.voiced = 0

    def current(self) -> np.ndarray:
        return np.concatenate(self.buf) if self.buf else np.zeros(0, dtype=np.float32)

    def push(self, frame: np.ndarray):
        p = self.model.prob(frame)
        if not self.in_speech:
            self.preroll.append(frame)
            self.preroll = self.preroll[-self.preroll_frames:]
            if p >= self.on:
                self.in_speech = True
                self.buf = list(self.preroll)
                self.probs = [0.0] * (len(self.buf) - 1) + [p]
                self.preroll = []
                self.quiet = 0
                self.voiced = 1
            return None

        self.buf.append(frame)
        self.probs.append(p)
        if p >= self.on:
            self.voiced += 1
        self.quiet = self.quiet + 1 if p < self.off else 0

        if self.quiet >= self.silence_frames:
            return self._emit(len(self.buf) - self.quiet + 3)
        if len(self.buf) >= self.max_frames:
            # too long without a pause: cut at the least speech-like frame in the last ~1.5 s
            tail = 47
            start = max(1, len(self.probs) - tail)
            cut = start + int(np.argmin(self.probs[start:])) + 1
            return self._emit(cut, keep_rest=True)
        return None

    def _emit(self, end: int, keep_rest: bool = False):
        seg, rest = self.buf[:end], self.buf[end:]
        rest_p = self.probs[end:]
        voiced = self.voiced
        self.buf, self.probs = [], []
        self.in_speech = False
        self.quiet = 0
        self.voiced = 0
        if keep_rest and rest:
            self.in_speech = True
            self.buf, self.probs = rest, rest_p
            self.voiced = sum(1 for x in rest_p if x >= self.on)
        else:
            self.model.reset()
        if voiced < self.min_frames or not seg:
            return None
        return np.concatenate(seg)
