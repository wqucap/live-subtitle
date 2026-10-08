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
    """Bring quiet audio up to a normal level so soft speech still trips the VAD.

    Tracks the RMS level (fast attack, slow release) and ramps the gain smoothly across each
    frame: a gain that jumps between frames distorts the signal and *lowers* Silero's
    confidence. Real captures can sit 30 dB below normal (player volume turned down), hence the
    high cap; digital near-silence below `floor_rms` is left alone."""

    def __init__(self, target_rms=0.08, max_gain=1000.0, floor_rms=3e-6, attack_s=0.05, release_s=2.0):
        self.target = target_rms
        self.max_gain = max_gain
        self.floor = floor_rms
        frame_s = FRAME / 16000
        self.a_up = 1 - np.exp(-frame_s / attack_s)
        self.a_down = 1 - np.exp(-frame_s / release_s)
        self.level = 0.0
        self.gain = 1.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(frame ** 2))) if len(frame) else 0.0
        self.level += (self.a_up if rms > self.level else self.a_down) * (rms - self.level)
        target = 1.0 if self.level < self.floor else min(self.max_gain, max(1.0, self.target / self.level))
        ramp = np.linspace(self.gain, target, len(frame), dtype=np.float32)
        self.gain = target
        return np.clip(frame * ramp, -1.0, 1.0)


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

    def open_with(self, frames: list[np.ndarray]):
        """Start an utterance from audio the VAD itself rejected (Whisper heard speech in it)."""
        self.in_speech = True
        self.buf = list(frames)
        self.probs = [1.0] * len(self.buf)
        self.preroll = []
        self.quiet = 0
        self.voiced = len(self.buf)

    def finish(self, sample: int | None = None):
        """Close the current utterance now (optionally at `sample`) and return it."""
        if not self.in_speech:
            return None
        end = len(self.buf) if sample is None else min(len(self.buf), max(1, round(sample / FRAME)))
        return self._emit(end)

    def push(self, frame: np.ndarray, force: bool = False):
        """force=True: treat the frame as speech whatever Silero says (Whisper-confirmed speech)."""
        p = self.model.prob(frame)
        if force:
            p = max(p, 1.0)
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

    def cut_at(self, sample: int):
        """Close the current utterance at `sample` (offset into current()) and keep the rest open.
        Used to finish a sentence as soon as Whisper sees it end, even without a pause."""
        if not self.in_speech:
            return None
        end = min(len(self.buf) - 1, max(1, round(sample / FRAME)))
        return self._emit(end, keep_rest=True)

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
