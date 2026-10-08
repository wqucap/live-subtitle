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
DIAG_DIR = ROOT / "diagnostics"


def cuda_dll_dirs() -> list[Path]:
    """Directories of the pip-installed nvidia-cublas-cu12 / nvidia-cudnn-cu12 DLLs
    (site-packages when run from source, _internal when frozen by PyInstaller)."""
    candidates = []
    search_roots = [Path(p) for p in sys.path if p] + [Path(getattr(sys, "_MEIPASS", ROOT))]
    for base in search_roots:
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin"):
            d = base / sub
            if d.is_dir():
                candidates.append(d)
    return list(dict.fromkeys(candidates))


def setup_cuda_dlls() -> None:
    """Make cuBLAS / cuDNN visible to CTranslate2 (faster-whisper) on Windows."""
    for d in cuda_dll_dirs():
        try:
            os.add_dll_directory(str(d))
        except OSError:
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
