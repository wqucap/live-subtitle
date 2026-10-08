"""Always-on-top translucent subtitle window."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QLabel, QSizeGrip, QVBoxLayout, QWidget


def _shadow(widget: QWidget) -> None:
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(6)
    eff.setOffset(1, 1)
    eff.setColor(QColor(0, 0, 0, 230))
    widget.setGraphicsEffect(eff)


class SubtitleOverlay(QWidget):
    def __init__(self, cfg):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.cfg = cfg
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("字幕")
        self.setMinimumSize(300, 80)
        self._drag: QPoint | None = None
        self.locked = False

        self.prev_label = QLabel("")
        self.zh_label = QLabel("字幕窗口：按住拖动，右下角可缩放")
        self.en_label = QLabel("")
        for lab in (self.prev_label, self.zh_label, self.en_label):
            lab.setWordWrap(True)
            lab.setAlignment(Qt.AlignCenter)
            lab.setAttribute(Qt.WA_TransparentForMouseEvents)
            _shadow(lab)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(2)
        lay.addWidget(self.prev_label)
        lay.addWidget(self.zh_label)
        lay.addWidget(self.en_label)
        self.grip = QSizeGrip(self)
        lay.addWidget(self.grip, 0, Qt.AlignBottom | Qt.AlignRight)

        self.apply_style()
        if cfg.overlay_geometry and len(cfg.overlay_geometry) == 4:
            self.setGeometry(*cfg.overlay_geometry)
        else:
            screen = self.screen().availableGeometry()
            w, h = int(screen.width() * 0.6), 150
            self.setGeometry(screen.x() + (screen.width() - w) // 2, screen.y() + screen.height() - h - 60, w, h)

    def apply_style(self):
        c = self.cfg
        f = QFont("Microsoft YaHei UI")
        f.setPixelSize(c.font_size)
        f.setBold(True)
        self.zh_label.setFont(f)
        self.zh_label.setStyleSheet("color: #ffffff;")
        f2 = QFont("Microsoft YaHei UI")
        f2.setPixelSize(max(10, int(c.font_size * 0.7)))
        self.prev_label.setFont(f2)
        self.prev_label.setStyleSheet("color: rgba(255,255,255,150);")
        f3 = QFont("Segoe UI")
        f3.setPixelSize(c.en_font_size)
        self.en_label.setFont(f3)
        self.en_label.setStyleSheet("color: #ffe08a;")
        self.en_label.setVisible(c.show_english)
        self.update()

    def set_locked(self, locked: bool):
        """Locked = clicks pass through to the game/video underneath."""
        self.locked = locked
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowTransparentForInput, locked)
        self.grip.setVisible(not locked)
        if visible:
            self.show()
        self.update()

    # --- content ---
    def show_partial(self, en: str):
        self.en_label.setText(en)

    def show_final(self, en: str):
        self.en_label.setText(en)

    def show_translation(self, zh: str, en: str):
        prev = self.zh_label.text()
        if prev and not prev.startswith("字幕窗口："):
            self.prev_label.setText(prev)
        self.zh_label.setText(zh)
        self.en_label.setText(en)

    def clear_text(self):
        self.prev_label.setText("")
        self.zh_label.setText("")
        self.en_label.setText("")

    # --- painting / dragging ---
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        alpha = int(255 * self.cfg.bg_opacity / 100)
        p.setBrush(QColor(0, 0, 0, alpha))
        if not self.locked:
            p.setPen(QColor(255, 255, 255, 60))
        else:
            p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 10, 10)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _):
        self._drag = None

    def closeEvent(self, e):
        # Alt+F4 on the subtitle only hides it; the control panel owns the real shutdown
        if not self.property("quitting"):
            e.ignore()
            self.hide()
            if self.on_hidden:
                self.on_hidden()
            return
        super().closeEvent(e)

    on_hidden = None

    def geometry_list(self):
        g = self.geometry()
        return [g.x(), g.y(), g.width(), g.height()]
