import json
from dataclasses import asdict, dataclass, field, fields

from paths import CONFIG_PATH


@dataclass
class Config:
    # 翻译模式: "local" = 视频模式(本地 Hunyuan-MT), "api" = 游戏模式(在线 API)
    mode: str = "local"

    # 语音识别
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "int8_float16"
    source_language: str = "en"
    audio_device: str = ""  # 空 = 系统默认输出设备

    # 断句
    vad_threshold: float = 0.5
    auto_gain: bool = True  # 自动放大轻声
    silence_ms: int = 500
    max_segment_s: float = 8.0
    show_partial: bool = True

    # 本地翻译 (llama.cpp)
    local_model_file: str = "Hunyuan-MT-7B.Q4_K_M.gguf"
    local_port: int = 18080
    local_gpu_layers: int = 99

    # 在线 API (OpenAI 兼容, 例如硅基流动)
    api_base_url: str = "https://api.siliconflow.cn/v1"
    api_key: str = ""
    api_model: str = "tencent/Hunyuan-MT-7B"

    target_language: str = "中文"

    # 字幕窗口
    font_size: int = 26
    zh_color: str = "#ffffff"
    en_font_size: int = 18
    en_color: str = "#ffe08a"
    show_english: bool = True
    bg_opacity: int = 55  # 0-100
    overlay_geometry: list = field(default_factory=lambda: [])  # x, y, w, h

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                names = {f.name for f in fields(cls)}
                for k, v in data.items():
                    if k in names:
                        setattr(cfg, k, v)
            except (OSError, ValueError):
                pass
        return cfg

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
