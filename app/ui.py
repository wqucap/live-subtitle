"""Control panel window."""
import datetime
import html
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QRadioButton, QSlider, QSpinBox, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

from audio import list_loopback_devices
from config import Config
from engine import Callbacks, Engine
from overlay import SubtitleOverlay

STATE_COLORS = {"off": "#9e9e9e", "loading": "#f0a020", "ok": "#2eb82e", "error": "#e04040"}

HELP_HTML = """
<h2>实时字幕翻译 · 使用说明</h2>
<h3>快速开始</h3>
<ol>
<li>在「主页」选择模式（第一次用推荐 <b>🎬 视频模式</b>）。</li>
<li>点 <b>▶ 开始翻译</b>。第一次会自动下载语音识别模型（约 1.6 GB）和翻译模型（约 4.3 GB），「运行状态」里能看到进度，之后秒开。</li>
<li>播放任何英文视频/语音，屏幕下方的字幕窗口就会显示中文。</li>
</ol>
<h3>两种模式</h3>
<ul>
<li><b>🎬 视频模式（本地）</b>：语音识别 + 翻译全部在你的显卡上运行，离线、免费，约占 7 GB 显存。</li>
<li><b>🎮 游戏模式（在线 API）</b>：翻译交给在线接口（默认硅基流动免费的 Hunyuan-MT-7B），只占约 2 GB 显存，几乎不影响游戏帧数。需要先在「设置」里填 API Key。</li>
</ul>
<h3>字幕窗口</h3>
<ul>
<li>按住字幕窗口任意位置拖动，右下角拖动可缩放。</li>
<li>中文、英文的字号和颜色可以分别调整（点颜色按钮选色），「恢复默认样式」一键还原。</li>
<li>勾选 <b>🔒 锁定字幕窗口</b> 后鼠标会穿透字幕，玩游戏不会误点；要移动时取消勾选。</li>
<li>游戏请用 <b>无边框窗口</b> 模式，独占全屏时看不到字幕。</li>
</ul>
<h3>声音来源</h3>
<p>翻译的是「电脑正在播放的声音」。如果你有多个输出设备（耳机/音箱），在「声音来源」里选你正在用的那个。</p>
<h3>小技巧</h3>
<ul>
<li>「历史记录」页可以回看所有翻译。</li>
<li>「测试翻译」可以手动输入英文检查翻译效果。</li>
<li>字幕延迟：对方说完一句后约 0.5–1 秒出现；「断句停顿」调小会更快，但句子容易被切碎。</li>
<li>轻声没被识别：在「设置」确认「增强轻声」已勾选，或把「识别灵敏度」调到「高」。</li>
<li>字幕窗口右上角的 × 可以关掉字幕；在主页勾选「显示字幕窗口」重新打开。</li>
</ul>
"""


class Bridge(QObject):
    status = Signal(str, str, str)
    partial = Signal(str)
    final = Signal(int, str)
    translated = Signal(int, str, str, float)
    error = Signal(str)
    test_result = Signal(str)


class StatusRow(QWidget):
    def __init__(self, title: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.dot = QLabel("●")
        self.title = QLabel(title)
        self.title.setFixedWidth(70)
        self.msg = QLabel("未加载")
        self.msg.setStyleSheet("color: #666;")
        lay.addWidget(self.dot)
        lay.addWidget(self.title)
        lay.addWidget(self.msg, 1)
        self.set("off", "未加载")

    def set(self, state: str, msg: str):
        self.dot.setStyleSheet(f"color: {STATE_COLORS.get(state, '#999')}; font-size: 16px;")
        self.msg.setText(msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = Config.load()
        self.setWindowTitle("实时字幕翻译")
        self.resize(560, 640)

        self.bridge = Bridge()
        self.bridge.status.connect(self._on_status)
        self.bridge.partial.connect(self._on_partial)
        self.bridge.final.connect(self._on_final)
        self.bridge.translated.connect(self._on_translated)
        self.bridge.error.connect(self._on_error)
        self.bridge.test_result.connect(lambda s: self.test_out.setText(s))

        self.engine = Engine(self.cfg, Callbacks(
            status=self.bridge.status.emit, partial=self.bridge.partial.emit, final=self.bridge.final.emit,
            translated=self.bridge.translated.emit, error=self.bridge.error.emit,
        ))
        self.overlay = SubtitleOverlay(self.cfg)
        self.overlay.show()

        tabs = QTabWidget()
        tabs.addTab(self._build_home(), "🏠 主页")
        tabs.addTab(self._build_history(), "📜 历史记录")
        tabs.addTab(self._build_settings(), "⚙ 设置")
        help_view = QTextBrowser()
        help_view.setHtml(HELP_HTML)
        tabs.addTab(help_view, "❓ 使用说明")
        self.setCentralWidget(tabs)

    # ------------------------------------------------------------------ pages
    def _build_home(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)

        mode_box = QGroupBox("① 选择模式")
        mv = QVBoxLayout(mode_box)
        self.rb_local = QRadioButton("🎬 视频模式 —— 全部本地运行，离线免费（约 7 GB 显存）")
        self.rb_api = QRadioButton("🎮 游戏模式 —— 翻译走在线 API，省显存（约 2 GB）")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_local)
        self.mode_group.addButton(self.rb_api)
        (self.rb_local if self.cfg.mode == "local" else self.rb_api).setChecked(True)
        mv.addWidget(self.rb_local)
        mv.addWidget(self.rb_api)
        v.addWidget(mode_box)

        src_box = QGroupBox("② 声音来源（翻译电脑正在播放的声音）")
        sh = QHBoxLayout(src_box)
        self.dev_combo = QComboBox()
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh_devices)
        sh.addWidget(self.dev_combo, 1)
        sh.addWidget(refresh)
        self._refresh_devices()
        v.addWidget(src_box)

        self.start_btn = QPushButton("▶  开始翻译")
        f = QFont()
        f.setPixelSize(18)
        f.setBold(True)
        self.start_btn.setFont(f)
        self.start_btn.setMinimumHeight(48)
        self.start_btn.clicked.connect(self._toggle)
        v.addWidget(self.start_btn)

        st_box = QGroupBox("③ 运行状态")
        sv = QVBoxLayout(st_box)
        self.rows = {"asr": StatusRow("语音识别"), "mt": StatusRow("翻译引擎"), "audio": StatusRow("声音")}
        for r in self.rows.values():
            sv.addWidget(r)
        self.latency = QLabel("延迟：—")
        self.latency.setStyleSheet("color: #666;")
        sv.addWidget(self.latency)
        v.addWidget(st_box)

        ov_box = QGroupBox("④ 字幕窗口")
        of = QFormLayout(ov_box)
        row = QHBoxLayout()
        self.cb_show = QCheckBox("显示字幕窗口")
        self.cb_show.setChecked(True)
        self.cb_show.toggled.connect(lambda on: self.overlay.setVisible(on))
        self.overlay.on_hidden = lambda: self.cb_show.setChecked(False)
        self.cb_lock = QCheckBox("🔒 锁定（鼠标穿透，玩游戏用）")
        self.cb_lock.toggled.connect(self.overlay.set_locked)
        row.addWidget(self.cb_show)
        row.addWidget(self.cb_lock)
        of.addRow(row)
        self.cb_en = QCheckBox("同时显示英文原文")
        self.cb_en.setChecked(self.cfg.show_english)
        self.cb_en.toggled.connect(self._style_changed)
        of.addRow(self.cb_en)
        self.sp_font = QSpinBox()
        self.sp_font.setRange(12, 96)
        self.sp_font.setSuffix(" px")
        self.sp_font.setValue(self.cfg.font_size)
        self.sp_font.valueChanged.connect(self._style_changed)
        self.btn_zh_color = QPushButton()
        self.btn_zh_color.clicked.connect(lambda: self._pick_color("zh_color", self.btn_zh_color))
        of.addRow("中文 大小 / 颜色", self._pair(self.sp_font, self.btn_zh_color))
        self.sp_en_font = QSpinBox()
        self.sp_en_font.setRange(10, 72)
        self.sp_en_font.setSuffix(" px")
        self.sp_en_font.setValue(self.cfg.en_font_size)
        self.sp_en_font.valueChanged.connect(self._style_changed)
        self.btn_en_color = QPushButton()
        self.btn_en_color.clicked.connect(lambda: self._pick_color("en_color", self.btn_en_color))
        of.addRow("英文 大小 / 颜色", self._pair(self.sp_en_font, self.btn_en_color))
        self._paint_color_buttons()
        reset = QPushButton("恢复默认样式")
        reset.clicked.connect(self._reset_style)
        of.addRow(reset)
        self.sl_opacity = QSlider(Qt.Horizontal)
        self.sl_opacity.setRange(0, 100)
        self.sl_opacity.setValue(self.cfg.bg_opacity)
        self.sl_opacity.valueChanged.connect(self._style_changed)
        of.addRow("背景不透明度", self.sl_opacity)
        v.addWidget(ov_box)
        v.addStretch(1)
        return page

    def _build_history(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        self.history = QTextBrowser()
        v.addWidget(self.history, 1)
        clear = QPushButton("清空记录")
        clear.clicked.connect(self.history.clear)
        v.addWidget(clear)

        test_box = QGroupBox("测试翻译（开始翻译后可用）")
        th = QVBoxLayout(test_box)
        row = QHBoxLayout()
        self.test_in = QLineEdit()
        self.test_in.setPlaceholderText("输入一句英文，回车翻译")
        self.test_in.returnPressed.connect(self._test_translate)
        go = QPushButton("翻译")
        go.clicked.connect(self._test_translate)
        row.addWidget(self.test_in, 1)
        row.addWidget(go)
        th.addLayout(row)
        self.test_out = QLabel("")
        self.test_out.setWordWrap(True)
        self.test_out.setTextInteractionFlags(Qt.TextSelectableByMouse)
        th.addWidget(self.test_out)
        v.addWidget(test_box)
        return page

    def _build_settings(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)

        asr = QGroupBox("语音识别 (Whisper)")
        af = QFormLayout(asr)
        self.cb_whisper = QComboBox()
        self.cb_whisper.addItems(["large-v3-turbo", "distil-large-v3", "medium.en", "small.en"])
        self.cb_whisper.setCurrentText(self.cfg.whisper_model)
        self.cb_whisper.setToolTip("large-v3-turbo 最准；small.en 最省显存")
        af.addRow("模型", self.cb_whisper)
        self.sp_silence = QSpinBox()
        self.sp_silence.setRange(200, 2000)
        self.sp_silence.setSingleStep(50)
        self.sp_silence.setSuffix(" 毫秒")
        self.sp_silence.setValue(self.cfg.silence_ms)
        self.sp_silence.setToolTip("对方停顿多久算一句话说完。越小字幕越快，但句子容易被切碎")
        af.addRow("断句停顿", self.sp_silence)
        self.cb_partial = QCheckBox("说话过程中先显示英文（实时预览）")
        self.cb_partial.setChecked(self.cfg.show_partial)
        af.addRow(self.cb_partial)
        self.cb_gain = QCheckBox("增强轻声（自动放大小音量的说话声）")
        self.cb_gain.setChecked(self.cfg.auto_gain)
        self.cb_gain.setToolTip("视频里小声说话、远处的人声也能识别。一般保持打开")
        af.addRow(self.cb_gain)
        self.cb_sens = QComboBox()
        for label, th in (("标准（推荐）", 0.5), ("高 —— 更容易捕捉轻声，偶尔会误识别噪音", 0.4),
                          ("低 —— 只识别清楚的人声，适合背景很吵的游戏", 0.6)):
            self.cb_sens.addItem(label, th)
        i = self.cb_sens.findData(self.cfg.vad_threshold)
        self.cb_sens.setCurrentIndex(max(0, i))
        af.addRow("识别灵敏度", self.cb_sens)
        v.addWidget(asr)

        api = QGroupBox("游戏模式 · 在线翻译 API（OpenAI 兼容接口）")
        pf = QFormLayout(api)
        self.ed_url = QLineEdit(self.cfg.api_base_url)
        self.ed_key = QLineEdit(self.cfg.api_key)
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.ed_key.setPlaceholderText("在硅基流动官网注册后获取")
        self.ed_model = QLineEdit(self.cfg.api_model)
        pf.addRow("接口地址", self.ed_url)
        pf.addRow("API Key", self.ed_key)
        pf.addRow("模型名", self.ed_model)
        hint = QLabel("默认：硅基流动 SiliconFlow 的免费模型 tencent/Hunyuan-MT-7B。\n"
                      "也可以换成阿里云百炼等其他兼容 OpenAI 格式的服务。")
        hint.setStyleSheet("color: #666;")
        pf.addRow(hint)
        v.addWidget(api)

        save = QPushButton("💾 保存设置（下次开始翻译时生效）")
        save.clicked.connect(self._save_settings)
        v.addWidget(save)
        v.addStretch(1)
        return page

    # ------------------------------------------------------------------ actions
    def _refresh_devices(self):
        self.dev_combo.clear()
        self.dev_combo.addItem("系统默认输出设备", "")
        try:
            for name in list_loopback_devices():
                self.dev_combo.addItem(name.replace(" [Loopback]", ""), name)
        except Exception as e:  # noqa: BLE001
            self._on_error(f"读取声音设备失败：{e}")
        i = self.dev_combo.findData(self.cfg.audio_device)
        self.dev_combo.setCurrentIndex(max(0, i))

    def _collect(self):
        c = self.cfg
        c.mode = "local" if self.rb_local.isChecked() else "api"
        c.audio_device = self.dev_combo.currentData() or ""
        c.whisper_model = self.cb_whisper.currentText()
        c.silence_ms = self.sp_silence.value()
        c.show_partial = self.cb_partial.isChecked()
        c.auto_gain = self.cb_gain.isChecked()
        c.vad_threshold = self.cb_sens.currentData()
        c.api_base_url = self.ed_url.text().strip()
        c.api_key = self.ed_key.text().strip()
        c.api_model = self.ed_model.text().strip()
        c.show_english = self.cb_en.isChecked()
        c.font_size = self.sp_font.value()
        c.en_font_size = self.sp_en_font.value()
        c.bg_opacity = self.sl_opacity.value()
        c.overlay_geometry = self.overlay.geometry_list()

    def _save_settings(self):
        self._collect()
        self.cfg.save()
        QMessageBox.information(self, "已保存", "设置已保存。正在运行时，请先停止再重新开始让新设置生效。")

    def _style_changed(self, *_):
        self.cfg.show_english = self.cb_en.isChecked()
        self.cfg.font_size = self.sp_font.value()
        self.cfg.en_font_size = self.sp_en_font.value()
        self.cfg.bg_opacity = self.sl_opacity.value()
        self.overlay.apply_style()

    @staticmethod
    def _pair(spin: QSpinBox, button: QPushButton) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(spin, 1)
        h.addWidget(button, 1)
        return w

    def _paint_color_buttons(self):
        for btn, color in ((self.btn_zh_color, self.cfg.zh_color), (self.btn_en_color, self.cfg.en_color)):
            text = "#000" if QColor(color).lightness() > 140 else "#fff"
            btn.setText(f"🎨 {color}")
            btn.setStyleSheet(f"background: {color}; color: {text}; border: 1px solid #888; padding: 4px;")

    def _pick_color(self, attr: str, btn: QPushButton):
        color = QColorDialog.getColor(QColor(getattr(self.cfg, attr)), self, "选择字幕颜色")
        if color.isValid():
            setattr(self.cfg, attr, color.name())
            self._paint_color_buttons()
            self.overlay.apply_style()

    def _reset_style(self):
        d = Config()
        self.cfg.zh_color, self.cfg.en_color = d.zh_color, d.en_color
        self.sp_font.setValue(d.font_size)
        self.sp_en_font.setValue(d.en_font_size)
        self.sl_opacity.setValue(d.bg_opacity)
        self._paint_color_buttons()
        self._style_changed()

    def _toggle(self):
        if self.engine.running:
            self.start_btn.setEnabled(False)
            threading.Thread(target=self._stop_engine, daemon=True).start()
            return
        self._collect()
        self.cfg.save()
        if self.cfg.mode == "api" and not self.cfg.api_key:
            QMessageBox.warning(self, "缺少 API Key", "游戏模式需要在「⚙ 设置」里填写 API Key。\n"
                                "没有的话先用 🎬 视频模式（本地）。")
            return
        self.start_btn.setText("■  停止翻译")
        self.rb_local.setEnabled(False)
        self.rb_api.setEnabled(False)
        self.engine.start()

    def _stop_engine(self):
        # switching modes frees the other model's VRAM, otherwise keep models warm for a fast restart
        self.engine.stop(unload=False)
        self.bridge.status.emit("_stopped", "", "")

    def _test_translate(self):
        text = self.test_in.text().strip()
        if not text:
            return
        self.test_out.setText("翻译中…")

        def run():
            try:
                self.bridge.test_result.emit(self.engine.translate_once(text))
            except Exception as e:  # noqa: BLE001
                self.bridge.test_result.emit(f"❌ {e}")

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------ engine events
    def _on_status(self, comp, state, msg):
        if comp == "_stopped":
            self.start_btn.setText("▶  开始翻译")
            self.start_btn.setEnabled(True)
            self.rb_local.setEnabled(True)
            self.rb_api.setEnabled(True)
            return
        if comp in self.rows:
            self.rows[comp].set(state, msg)

    def _on_partial(self, en):
        self.overlay.show_partial(en)

    def _on_final(self, _id, en):
        self.overlay.show_final(en)

    def _on_translated(self, _id, en, zh, secs):
        self.overlay.show_translation(zh, en)
        self.latency.setText(f"延迟：说完后 {secs:.2f} 秒出字幕")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.history.append(
            f"<span style='color:#999'>{ts}</span> <b>{html.escape(zh)}</b><br>"
            f"<span style='color:#777'>{html.escape(en)}</span><br>")

    def _on_error(self, msg):
        if not self.engine.running:
            self.bridge.status.emit("_stopped", "", "")
        for r in self.rows.values():
            if r.msg.text().endswith("…"):
                r.set("error", "失败")
        QMessageBox.warning(self, "出错了", msg)

    def closeEvent(self, e):
        self._collect()
        self.cfg.save()
        self.engine.stop(unload=True)
        self.overlay.setProperty("quitting", True)
        self.overlay.close()
        super().closeEvent(e)
