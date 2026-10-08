"""Pipeline: loopback audio -> VAD segmentation -> faster-whisper -> translator.

Runs in background threads and reports through plain callbacks (the UI wraps them in Qt signals)."""
import collections
import datetime
import logging
import queue
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from audio import TARGET_RATE, LoopbackCapture
from config import Config
from translator import LlamaServer, Translator, ensure_local_model
from paths import DIAG_DIR, WHISPER_DIR
from vad import FRAME, AutoGain, SpeechDetector

log = logging.getLogger(__name__)

# Whisper likes to invent these on music / silence.
HALLUCINATIONS = re.compile(
    r"^(thanks? (you )?(so much )?for watching|thank you\.?|you|bye\.?|please subscribe|"
    r"subtitles by .*|\.+|♪+)[.!]?$",
    re.IGNORECASE,
)


ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "jr.", "sr."}


def _sentence_cut(words, audio_s: float, clause: bool = False, min_s: float = 1.0, tail_s: float = 0.8):
    """Sample offset just after the last sentence-final word, if the sentence is safely over.
    Whisper tends to put a period after whatever word is last in an unfinished buffer, so a
    boundary only counts with at least two words after it and some distance from the buffer end.
    clause=True also accepts , ; : (used when the length cap is near, to avoid a mid-phrase cut)."""
    marks = ".?!,;:" if clause else ".?!"
    cut = None
    for i, w in enumerate(words[:-2]):
        token = w.word.strip()
        if (i >= 2  # at least 3 words, so a lone "Inside." / "Well." doesn't become its own subtitle
                and token[-1:] in marks and token.lower() not in ABBREVIATIONS
                and min_s <= w.end <= audio_s - tail_s):
            nxt = words[i + 1]
            cut = int((w.end + min(nxt.start, w.end + 0.3)) / 2 * TARGET_RATE)
    return cut


@dataclass
class Heard:
    text: str
    cut: int | None  # sample where a finished sentence ends with more speech after it
    last_end: float  # seconds: end of the last recognised word
    n_words: int
    confident: bool  # strict: clearly speech, not music/noise


@dataclass
class Callbacks:
    status: Callable[[str, str, str], None]  # (component, state: off|loading|ok|error, message)
    partial: Callable[[str], None]  # interim English while someone is still talking
    final: Callable[[int, str], None]  # (segment id, English)
    translated: Callable[[int, str, str, float], None]  # (segment id, English, translation, seconds)
    error: Callable[[str], None]
    partial_translated: Callable[[str], None] = lambda zh: None  # interim Chinese while still talking


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
        # interim translation: only the newest partial matters, and only while its utterance is still open
        self._utt = 0
        self._partial_lock = threading.Lock()
        self._partial_job: tuple[int, str] | None = None
        # last ~30 s of captured audio, kept in memory only; written to disk when the user asks
        self._recent: "collections.deque[np.ndarray]" = collections.deque()
        self._recent_samples = 0
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
    def _transcribe(self, audio: np.ndarray, beam: int, words: bool = False, clause: bool = False):
        """Returns the text, or a Heard (with word timing, sentence cut and confidence) when words=True."""
        # normalise each utterance so quiet speech reaches Whisper at a healthy level
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1e-4:
            audio = audio * (0.9 / peak)
        with self._asr_lock:
            segments, _ = self._whisper.transcribe(
                audio, language=self.cfg.source_language or None, beam_size=beam,
                condition_on_previous_text=False, vad_filter=False,
                without_timestamps=not words, word_timestamps=words,
            )
            parts, word_list, kept = [], [], []
            for s in segments:
                if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                    if beam > 1:  # final pass only; interim passes run every 0.8 s
                        log.info("dropped low-confidence text (no_speech=%.2f logprob=%.2f): %s",
                                 s.no_speech_prob, s.avg_logprob, s.text.strip())
                    continue
                parts.append(s.text.strip())
                word_list.extend(s.words or [])
                kept.append(s)
        text = " ".join(p for p in parts if p).strip()
        if HALLUCINATIONS.match(text):
            if beam > 1:
                log.info("dropped likely hallucination: %s", text)
            text = ""
        if not words:
            return text
        # strict bar for "this really is speech" (used to override the VAD)
        confident = bool(text) and len(word_list) >= 3 and all(
            s.no_speech_prob < 0.3 and s.avg_logprob > -0.5 and s.compression_ratio < 2.4 for s in kept)
        return Heard(
            text=text,
            cut=_sentence_cut(word_list, len(audio) / TARGET_RATE, clause) if text else None,
            last_end=word_list[-1].end if (text and word_list) else 0.0,
            n_words=len(word_list) if text else 0,
            confident=confident,
        )

    def _audio_loop(self):
        vad = SpeechDetector(
            threshold=self.cfg.vad_threshold, silence_ms=self.cfg.silence_ms,
            max_segment_s=self.cfg.max_segment_s,
        )
        agc = AutoGain() if self.cfg.auto_gain else None
        pending = np.zeros(0, dtype=np.float32)
        last_partial = 0.0
        # Whisper fallback: audio the VAD called "not speech" is kept here (last 3 s) and Whisper
        # listens to it every 1.5 s. Silero misses e.g. tape-recorder / phone audio that Whisper
        # transcribes with high confidence; when that happens we follow Whisper ("rescue" mode).
        idle: "collections.deque[np.ndarray]" = collections.deque(maxlen=int(3.0 * TARGET_RATE / FRAME))
        last_probe = 0.0
        rescue = False
        while not self._stop.is_set():
            try:
                chunk = self._capture.queue.get(timeout=0.2) if self._capture else None
            except queue.Empty:
                chunk = None
            if chunk is not None:
                self._remember(chunk)
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
                segment = vad.push(frame, force=rescue)
                if segment is not None:
                    self._finish_segment(segment)
                if vad.in_speech:
                    idle.clear()
                else:
                    idle.append(frame)
                    rescue = False

            if (self.cfg.whisper_rescue and not vad.in_speech and len(idle) * FRAME >= 1.5 * TARGET_RATE
                    and time.time() - last_probe > 1.5 and not self._asr_lock.locked()):
                last_probe = time.time()
                window = np.concatenate(idle)
                if np.sqrt(np.mean(window ** 2)) > 0.003:  # skip (near-)silence
                    try:
                        heard = self._transcribe(window, beam=1, words=True)
                    except Exception:  # noqa: BLE001
                        log.exception("rescue probe failed")
                        heard = None
                    if heard and heard.confident:
                        log.info("VAD missed speech, Whisper heard %d words: following Whisper", heard.n_words)
                        vad.open_with(list(idle))
                        idle.clear()
                        rescue = True
                        last_partial = 0.0

            interim = self.cfg.show_partial or self.cfg.sentence_split or rescue
            if interim and vad.in_speech and time.time() - last_partial > 0.8:
                buf = vad.current()
                if len(buf) > TARGET_RATE * 0.6 and not self._asr_lock.locked():
                    last_partial = time.time()
                    try:
                        if self.cfg.sentence_split or rescue:
                            near_cap = len(buf) / TARGET_RATE >= self.cfg.max_segment_s - 1.5
                            heard = self._transcribe(buf, beam=1, words=True, clause=near_cap)
                            text, cut = heard.text, (heard.cut if self.cfg.sentence_split else None)
                        else:
                            heard, text, cut = None, self._transcribe(buf, beam=1), None
                    except Exception:  # noqa: BLE001
                        log.exception("partial transcribe failed")
                        continue
                    if rescue:
                        # the VAD can't tell when this kind of speech stops, so Whisper decides:
                        # nothing recognised, or no word in the last 1.2 s -> close the utterance
                        if not text:
                            vad.finish()
                            rescue = False
                            continue
                        if len(buf) / TARGET_RATE - heard.last_end > 1.2:
                            segment = vad.finish(int((heard.last_end + 0.3) * TARGET_RATE))
                            rescue = False
                            if segment is not None:
                                self._finish_segment(segment)
                            continue
                    if cut is not None:
                        # a sentence already ended mid-stream (no pause, e.g. over background music):
                        # finalise it now instead of waiting for silence or the length cap
                        segment = vad.cut_at(cut)
                        if segment is not None:
                            self._finish_segment(segment)
                        continue
                    if text and self.cfg.show_partial:
                        self.cb.partial(text)
                        if self.cfg.partial_translate:
                            with self._partial_lock:
                                self._partial_job = (self._utt, text)

    def _finish_segment(self, audio: np.ndarray):
        t0 = time.time()
        with self._partial_lock:
            self._utt += 1  # interim translations of this utterance are now stale
            self._partial_job = None
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
        last_partial_src = ""
        while not self._stop.is_set():
            try:
                item = self._tq.get(timeout=0.05)
            except queue.Empty:
                last_partial_src = self._translate_partial(last_partial_src)
                continue
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

    def _translate_partial(self, previous: str) -> str:
        """Translate the newest interim English, if it changed. Only runs when no finished
        sentence is waiting, so final translations always go first. Returns the text translated."""
        with self._partial_lock:
            job, self._partial_job = self._partial_job, None
        if not job or job[1] == previous:
            return previous
        utt, text = job
        try:
            zh = self._translator.translate(text, fragment=True)
        except Exception:  # noqa: BLE001 - a dropped preview is harmless; the final pass reports errors
            log.exception("partial translate failed")
            return previous
        with self._partial_lock:
            if utt != self._utt:
                return text  # sentence finished meanwhile; its final translation is on the way
        self.cb.partial_translated(zh)
        return text

    # ---------- diagnostics ----------
    RECENT_S = 30

    def _remember(self, chunk: np.ndarray):
        self._recent.append(chunk)
        self._recent_samples += len(chunk)
        while self._recent_samples - len(self._recent[0]) >= self.RECENT_S * TARGET_RATE:
            self._recent_samples -= len(self._recent.popleft())

    def save_recent(self) -> Path:
        """Write the last ~30 s of what was heard to diagnostics/ (only when the user asks)."""
        chunks = list(self._recent)
        if not chunks:
            raise RuntimeError("还没有录到声音（需要先开始翻译并播放视频）")
        audio = np.concatenate(chunks)
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        path = DIAG_DIR / f"missed-{datetime.datetime.now():%Y%m%d-%H%M%S}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        log.info("saved %.1fs of recent audio to %s (settings: vad=%.2f gain=%s split=%s)",
                 len(audio) / TARGET_RATE, path, self.cfg.vad_threshold, self.cfg.auto_gain,
                 self.cfg.sentence_split)
        return path

    def translate_once(self, text: str) -> str:
        if not self._translator:
            raise RuntimeError("翻译引擎还没启动")
        return self._translator.translate(text)
