import logging
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from paths import LOG_PATH, ROOT, setup_cuda_dlls  # noqa: E402

setup_cuda_dlls()

from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui import MainWindow  # noqa: E402


def selftest(seconds: float) -> None:
    """`LiveSubtitle.exe --selftest 20`: run the engine without UI and log everything it produces."""
    import time

    from config import Config
    from engine import Callbacks, Engine

    log = logging.getLogger("selftest")
    e = Engine(Config.load(), Callbacks(
        status=lambda c, s, m: log.info("status %s %s %s", c, s, m),
        partial=lambda t: None,
        final=lambda i, t: log.info("final %d %s", i, t),
        translated=lambda i, en, zh, s: log.info("translated %d (%.2fs) %s", i, s, zh),
        error=lambda m: log.error("error %s", m),
    ))
    e.start()
    time.sleep(seconds)
    e.stop(unload=True)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8")],
    )
    if "--selftest" in sys.argv:
        selftest(float(sys.argv[sys.argv.index("--selftest") + 1]))
        return
    app = QApplication(sys.argv)
    app.setApplicationName("实时字幕翻译")
    icon = os.path.join(getattr(sys, "_MEIPASS", str(ROOT / "app")), "icon.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
