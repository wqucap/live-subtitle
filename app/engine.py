"""Pipeline: loopback audio -> VAD segmentation -> faster-whisper -> translator.

Runs in background threads and reports through plain callbacks (the UI wraps them in Qt signals)."""
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from audio import TARGET_RATE, LoopbackCapture
from config import Config
from translator import LlamaServer, Translator, ensure_local_model
from paths import WHISPER_DIR
from vad import FRAME, AutoGain, SpeechDetector

log = logging.getLogger(__name__)

# Whisper likes to invent these on music / silence.
HALLUCINATIONS = re.compile(
    r"^(thanks? (you )?(so much )?for watching|thank you\.?|you|bye\.?|please subscribe|"
    r"subtitles by .*|\.+|♪+)[.!]?$",
    re.IGNORECASE,
)


@dataclass
class Callbacks:
    status: Callable[[str, str, str], None]  # (component, state: off|loading|ok|error, message)
    partial: Callable[[str], None]  # interim English while someone is still talking
    final: Callable[[int, str], None]  # (segment id, English)
    translated: Callable[[int, str, str, float], None]  # (segment id, English, translation, seconds)
    error: Callable[[str], None]


class Engine:
    def __init__(self, cfg: Config, cb: Callbacks):
        self.cfg = cfg
        self.cb = cb
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._whisper = None
        self._whisper_key = None
        self._asr_lock = threading.Lock()
        self._server: LlamaServer | None = None
        self._translator: Translator | None = None
        self._capture: LoopbackCapture | None = None
        self._tq: "queue.Queue[tuple[int, str, float] | None]" = queue.Queue()
        self._seg_id = 0
        self.running = False

    # ---------- model loading ----------
    def _load_whisper(self):
        key = (self.cfg.whisper_model, self.cfg.whisper_compute_type)
        if self._whisper is not None and self._whisper_key == key:
            return
        from faster_whisper import WhisperModel

        # Flat models/whisper/<name>/ instead of the deep HF cache layout, which can exceed MAX_PATH
        model_dir = WHISPER_DIR / self.cfg.whisper_model
        if not (model_dir / "model.bin").exists():
            self.cb.status("asr", "loading", f"首次使用，下载语音识别模型 {self.cfg.whisper_model}（约 1.6 GB）…")
            from faster_whisper.utils import _MODELS
            from huggingface_hub import snapshot_download

            snapshot_download(
                _MODELS[self.cfg.whisper_model], local_dir=str(model_dir),
                allow_patterns=["config.json", "preprocessor_config.json", "model.bin",
                                "tokenizer.json", "vocabulary.*"],
            )
        self.cb.status("asr", "loading", f"加载 Whisper {self.cfg.whisper_model}…")
        self._whisper = None
        self._whisper = WhisperModel(
            str(model_dir), device="cuda", compute_type=self.cfg.whisper_compute_type)
        self._whisper_key = key
        self.cb.status("asr", "ok", f"Whisper {self.cfg.whisper_model} 已就绪")

    def _load_translator(self):
        if self._translator:
            self._translator.close()
            self._translator = None
        if self.cfg.mode == "local":
            ensure_local_model(self.cfg.local_model_file, lambda done, total: self.cb.status(
                "mt", "loading",
                f"首次使用，下载翻译模型 {done / 2**30:.2f} / {total / 2**30:.2f} GB（{done * 100 // max(total, 1)}%）…"))
            self.cb.status("mt", "loading", "启动本地翻译模型 Hunyuan-MT-7B…")
            if self._server is None:
                self._server = LlamaServer(self.cfg.local_model_file, self.cfg.local_port, self.cfg.local_gpu_layers)
            self._server.start()
            self._translator = Translator(self._server.base_url, "hunyuan-mt", target=self.cfg.target_language)
            self.cb.status("mt", "ok", "本地 Hunyuan-MT-7B 已就绪")
        else:
            self._stop_server()
            if not self.cfg.api_key:
                raise RuntimeError("游戏模式需要先在设置里填写 API Key")
            self._translator = Translator(
                self.cfg.api_base_url, self.cfg.api_model, self.cfg.api_key, self.cfg.target_language)
            self.cb.status("mt", "ok", f"在线翻译：{self.cfg.api_model}")

    def _stop_server(self):
        if self._server:
            self._server.stop()
            self._server = None

    # ---------- lifecycle ----------
    def start(self):
        if self.running:
            return
        self._stop.clear()
        self.running = True
        t = threading.Thread(target=self._startup, daemon=True)
        t.start()

    def _startup(self):
        try:
            self._load_whisper()
            self._load_translator()
            self.cb.status("audio", "loading", "打开系统声音…")
            self._capture = LoopbackCapture(self.cfg.audio_device)
            self._capture.start()
            self.cb.status("audio", "ok", f"正在收听：{self._capture.device_label}")
        except Exception as e:  # noqa: BLE001 - surface any startup failure to the user
            log.exception("startup failed")
            self.cb.error(str(e))
            self.stop()
            return
        self._threads = [
            threading.Thread(target=self._audio_loop, daemon=True),
            threading.Thread(target=self._translate_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self, unload: bool = False):
        self._stop.set()
        self._tq.put(None)
        if self._capture:
            self._capture.stop()
            self._capture = None
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=3)
        self._threads = []
        self._tq = queue.Queue()
        self.running = False
        self.cb.status("audio", "off", "已停止")
        if unload:
            self._stop_server()
            self._whisper = None
            self.cb.status("mt", "off", "未加载")
            self.cb.status("asr", "off", "未加载")

    # ---------- workers ----------
    def _transcribe(self, audio: np.ndarray, beam: int) -> str:
        # normalise each utterance so quiet speech reaches Whisper at a healthy level
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1e-4:
            audio = audio * (0.9 / peak)
        with self._asr_lock:
            segments, _ = self._whisper.transcribe(
                audio, language=self.cfg.source_language or None, beam_size=beam,
                condition_on_previous_text=False, vad_filter=False,
                without_timestamps=True,
            )
            parts = []
            for s in segments:
                if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                    continue
                parts.append(s.text.strip())
        text = " ".join(p for p in parts if p).strip()
        if HALLUCINATIONS.match(text):
            return ""
        return text

    def _audio_loop(self):
        vad = SpeechDetector(
            threshold=self.cfg.vad_threshold, silence_ms=self.cfg.silence_ms,
            max_segment_s=self.cfg.max_segment_s,
        )
        agc = AutoGain() if self.cfg.auto_gain else None
        pending = np.zeros(0, dtype=np.float32)
        last_partial = 0.0
        while not self._stop.is_set():
            try:
                chunk = self._capture.queue.get(timeout=0.2) if self._capture else None
            except queue.Empty:
                chunk = None
            if chunk is None:
                # no audio callbacks while nothing plays -> treat as silence so open segments close
                chunk = np.zeros(int(TARGET_RATE * 0.2), dtype=np.float32) if vad.in_speech else None
                if chunk is None:
                    continue
            pending = np.concatenate([pending, chunk])
            while len(pending) >= FRAME:
                frame, pending = pending[:FRAME], pending[FRAME:]
                if agc:
                    frame = agc.process(frame)
                segment = vad.push(frame)
                if segment is not None:
                    self._finish_segment(segment)
            if self.cfg.show_partial and vad.in_speech and time.time() - last_partial > 0.8:
                buf = vad.current()
                if len(buf) > TARGET_RATE * 0.6 and not self._asr_lock.locked():
                    last_partial = time.time()
                    try:
                        text = self._transcribe(buf, beam=1)
                        if text:
                            self.cb.partial(text)
                    except Exception:  # noqa: BLE001
                        log.exception("partial transcribe failed")

    def _finish_segment(self, audio: np.ndarray):
        t0 = time.time()
        try:
            text = self._transcribe(audio, beam=3)
        except Exception as e:  # noqa: BLE001
            log.exception("transcribe failed")
            self.cb.error(f"语音识别出错：{e}")
            return
        if not text:
            self.cb.partial("")
            return
        self._seg_id += 1
        self.cb.final(self._seg_id, text)
        self._tq.put((self._seg_id, text, t0))

    def _translate_loop(self):
        while not self._stop.is_set():
            item = self._tq.get()
            if item is None:
                return
            seg_id, text, t0 = item
            try:
                zh = self._translator.translate(text)
            except Exception as e:  # noqa: BLE001
                log.exception("translate failed")
                self.cb.error(f"翻译出错：{e}")
                continue
            self.cb.translated(seg_id, text, zh, time.time() - t0)

    def translate_once(self, text: str) -> str:
        if not self._translator:
            raise RuntimeError("翻译引擎还没启动")
        return self._translator.translate(text)
