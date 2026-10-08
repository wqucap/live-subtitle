"""Analyse a diagnostics/missed-*.wav recording stage by stage to see where speech was lost.

    .venv\\Scripts\\python.exe tools\\diagnose.py diagnostics\\missed-20261008-213000.wav

Prints, per second: audio level and Silero speech probability (raw and after auto-gain);
then the utterances the VAD cuts with the app's settings, what Whisper hears in each and
whether the low-confidence filter would drop it; finally Whisper on the whole clip with no VAD.
"""
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import paths  # noqa: E402

paths.setup_cuda_dlls()
import numpy as np  # noqa: E402

import vad as V  # noqa: E402
from config import Config  # noqa: E402

SR = 16000


def main(path: str):
    cfg = Config.load()
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    print(f"{path}: {len(x) / SR:.1f}s, peak {np.abs(x).max():.3f}, "
          f"settings vad={cfg.vad_threshold} auto_gain={cfg.auto_gain}")

    raw_vad, gain_vad, agc = V._StreamingSilero(), V._StreamingSilero(), V.AutoGain()
    frames = range(0, len(x) - V.FRAME, V.FRAME)
    p_raw, p_gain = [], []
    for i in frames:
        f = x[i:i + V.FRAME]
        p_raw.append(raw_vad.prob(f))
        p_gain.append(gain_vad.prob(agc.process(f)))
    per_s = int(SR / V.FRAME)
    print("\n sec   level(dBFS)  speech-prob raw / with gain   (threshold %.2f)" % cfg.vad_threshold)
    for s in range(0, len(p_raw), per_s):
        seg = x[s * V.FRAME:(s + per_s) * V.FRAME]
        db = 20 * np.log10(max(1e-6, np.sqrt(np.mean(seg ** 2))))
        pr, pg = max(p_raw[s:s + per_s]), max(p_gain[s:s + per_s])
        bar = "#" * int(pg * 20)
        print(f" {s // per_s:3d}   {db:7.1f}      {pr:.2f} / {pg:.2f}  {bar}")

    from faster_whisper import WhisperModel
    m = WhisperModel(str(paths.WHISPER_DIR / cfg.whisper_model), device="cuda",
                     compute_type=cfg.whisper_compute_type)

    det = V.SpeechDetector(threshold=cfg.vad_threshold, silence_ms=cfg.silence_ms,
                           max_segment_s=cfg.max_segment_s)
    agc = V.AutoGain() if cfg.auto_gain else None
    print("\nutterances cut by the VAD with these settings:")
    found = 0
    for i in frames:
        f = x[i:i + V.FRAME]
        seg = det.push(agc.process(f) if agc else f)
        if seg is None:
            continue
        found += 1
        end = (i + V.FRAME) / SR
        seg = seg / max(1e-4, np.abs(seg).max()) * 0.9
        out, _ = m.transcribe(seg, language=cfg.source_language or None, beam_size=3,
                              condition_on_previous_text=False, without_timestamps=True)
        for z in out:
            drop = z.no_speech_prob > 0.6 and z.avg_logprob < -1.0
            print(f"  ends {end:5.1f}s len {len(seg) / SR:4.1f}s  no_speech={z.no_speech_prob:.2f} "
                  f"logprob={z.avg_logprob:.2f} {'[DROPPED] ' if drop else ''}| {z.text.strip()}")
    if not found:
        print("  (none — the VAD never opened an utterance)")

    print("\nWhisper on the whole clip, no VAD:")
    xx = x / max(1e-4, np.abs(x).max()) * 0.9
    out, _ = m.transcribe(xx, language=cfg.source_language or None, beam_size=3,
                          condition_on_previous_text=False, vad_filter=False)
    for z in out:
        print(f"  {z.start:5.1f}-{z.end:5.1f}s no_speech={z.no_speech_prob:.2f} "
              f"logprob={z.avg_logprob:.2f} | {z.text.strip()}")


if __name__ == "__main__":
    main(sys.argv[1])
