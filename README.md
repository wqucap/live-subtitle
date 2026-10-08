# 实时字幕翻译 (Live Subtitle)

看英文视频、打游戏时，把电脑正在播放的英文语音实时翻译成中文，以悬浮字幕显示在屏幕上。

- **🎬 视频模式**：语音识别（Whisper large-v3-turbo）+ 翻译（Hunyuan-MT-7B）全部在本地显卡运行，离线免费，约 7 GB 显存
- **🎮 游戏模式**：翻译改用在线 API（默认硅基流动免费的 `tencent/Hunyuan-MT-7B`，任何 OpenAI 兼容接口都行），约 2 GB 显存，几乎不影响游戏
- 说完一句话后约 0.2–0.5 秒出中文；说话过程中先显示英文预览
- 字幕窗口可拖动、缩放、调字号和透明度，可「锁定」让鼠标穿透

## 运行要求

- Windows 10/11，NVIDIA 显卡（建议 8 GB 显存以上，在 RTX 4080 SUPER 上测试）
- 游戏需用「无边框窗口」模式，字幕窗口才能显示在游戏上层

## 从源码安装

需要 [python.org](https://www.python.org/) 的 Python 3.12（不要用 Anaconda 的 Python，其自带的 VC++ 运行库与 PySide6 冲突）。

```powershell
.\setup.ps1          # 建 venv、装依赖、下载 llama.cpp 和翻译模型
.\.venv\Scripts\python.exe app\main.py
```

打包成 exe：

```powershell
.\build.ps1          # 生成 LiveSubtitle.exe + _internal\，与 bin\、models\ 放在同一目录
```

## 目录结构

```
app/
  main.py         入口（--selftest N 可无界面自检 N 秒，结果写入 live-subtitle.log）
  ui.py           控制面板
  overlay.py      悬浮字幕窗口
  engine.py       流水线：声音 → 断句 → 识别 → 翻译
  audio.py        WASAPI 回环录音（录系统正在播放的声音）
  vad.py          Silero VAD 流式断句
  translator.py   翻译（本地 llama-server / 在线 API）
bin/llama/        llama.cpp CUDA 版（setup.ps1 下载）
models/           翻译模型 GGUF + Whisper 模型（首次运行自动下载）
```

## 致谢

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) ·
[llama.cpp](https://github.com/ggml-org/llama.cpp) ·
[Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B) ·
[PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) ·
[PySide6](https://doc.qt.io/qtforpython-6/)
