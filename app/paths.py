"""Locate the app root (works both from source and from a PyInstaller exe) and wire up CUDA DLLs."""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
BIN_DIR = ROOT / "bin"
LLAMA_DIR = BIN_DIR / "llama"
CONFIG_PATH = ROOT / "config.json"
LOG_PATH = ROOT / "live-subtitle.log"


def setup_cuda_dlls() -> None:
    """Make cuBLAS / cuDNN visible to CTranslate2 (faster-whisper) on Windows."""
    candidates = []
    # pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 (source run, or collected into _internal by PyInstaller)
    search_roots = [Path(p) for p in sys.path if p] + [Path(getattr(sys, "_MEIPASS", ROOT))]
    for base in search_roots:
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin"):
            d = base / sub
            if d.is_dir():
                candidates.append(d)
    for d in dict.fromkeys(candidates):
        try:
            os.add_dll_directory(str(d))
        except OSError:
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
