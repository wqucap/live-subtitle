"""Capture what the speakers are playing (WASAPI loopback) as 16 kHz mono float32."""
import queue

import numpy as np
import pyaudiowpatch as pyaudio

TARGET_RATE = 16000


def list_loopback_devices() -> list[str]:
    p = pyaudio.PyAudio()
    try:
        return [d["name"] for d in p.get_loopback_device_info_generator()]
    finally:
        p.terminate()


def _resample(x: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == TARGET_RATE:
        return x
    if src_rate % TARGET_RATE == 0:
        # integer factor (48k/32k): box-filter + decimate, good enough for speech
        f = src_rate // TARGET_RATE
        n = len(x) // f * f
        return x[:n].reshape(-1, f).mean(axis=1)
    n_out = int(len(x) * TARGET_RATE / src_rate)
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


class LoopbackCapture:
    def __init__(self, device_name: str = ""):
        self.device_name = device_name
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._pa = None
        self._stream = None
        self.device_label = ""

    def _find_device(self, p):
        if self.device_name:
            for d in p.get_loopback_device_info_generator():
                if d["name"] == self.device_name:
                    return d
        return p.get_default_wasapi_loopback()

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        dev = self._find_device(self._pa)
        self.device_label = dev["name"]
        channels = int(dev["maxInputChannels"])
        rate = int(dev["defaultSampleRate"])
        carry = np.zeros(0, dtype=np.float32)

        def callback(in_data, frame_count, time_info, status):
            nonlocal carry
            x = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels).mean(axis=1)
            x = np.concatenate([carry, x])
            if rate % TARGET_RATE == 0:
                f = rate // TARGET_RATE
                keep = len(x) // f * f
                carry = x[keep:]
                x = x[:keep]
            self.queue.put(_resample(x, rate).astype(np.float32))
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32, channels=channels, rate=rate,
            input=True, input_device_index=dev["index"],
            frames_per_buffer=int(rate * 0.05), stream_callback=callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
