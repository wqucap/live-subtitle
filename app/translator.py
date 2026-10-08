"""Translation backends. Both talk to an OpenAI-compatible /chat/completions endpoint:
local = llama.cpp `llama-server` running Hunyuan-MT-7B, api = any online provider (e.g. SiliconFlow)."""
import subprocess
import time

import httpx

from paths import LLAMA_DIR, MODELS_DIR

# Prompt format recommended by the Hunyuan-MT model card for XX -> ZH.
PROMPT_ZH = "把下面的文本翻译成{lang}，不要额外解释。\n\n{text}"
PROMPT_OTHER = "Translate the following segment into {lang}, without additional explanation.\n\n{text}"


def build_prompt(text: str, target: str) -> str:
    tpl = PROMPT_ZH if target in ("中文", "简体中文", "繁体中文", "粤语") else PROMPT_OTHER
    return tpl.format(lang=target, text=text)


class Translator:
    def __init__(self, base_url: str, model: str, api_key: str = "", target: str = "中文"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.target = target
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(headers=headers, timeout=httpx.Timeout(30.0, connect=5.0))

    def translate(self, text: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_prompt(text, self.target)}],
            # sampling settings recommended for Hunyuan-MT, slightly cooler for stable subtitles
            "temperature": 0.3,
            "top_p": 0.6,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_tokens": 256,
            "stream": False,
        }
        r = self.client.post(f"{self.base_url}/chat/completions", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"翻译接口返回 {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        self.client.close()


class LlamaServer:
    """Runs bin/llama/llama-server.exe as a hidden child process."""

    def __init__(self, model_file: str, port: int, gpu_layers: int):
        self.model_path = MODELS_DIR / model_file
        self.port = port
        self.gpu_layers = gpu_layers
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self, timeout: float = 120.0) -> None:
        exe = LLAMA_DIR / "llama-server.exe"
        if not exe.exists():
            raise FileNotFoundError(f"找不到 {exe}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到翻译模型 {self.model_path}")
        if self._healthy():
            return  # already running (e.g. left over from a previous session)
        cmd = [
            str(exe), "-m", str(self.model_path),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-ngl", str(self.gpu_layers), "-c", "4096", "-np", "1",
            "--no-webui",
        ]
        self.proc = subprocess.Popen(
            cmd, cwd=str(LLAMA_DIR),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server 启动失败 (退出码 {self.proc.returncode})")
            if self._healthy():
                return
            time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama-server 启动超时")

    def _healthy(self) -> bool:
        try:
            return httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
