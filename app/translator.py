"""Translation backends. Both talk to an OpenAI-compatible /chat/completions endpoint:
local = llama.cpp `llama-server` running Hunyuan-MT-7B, api = any online provider (e.g. SiliconFlow)."""
import os
import re
import subprocess
import time

import httpx

from paths import LLAMA_DIR, MODELS_DIR, cuda_dll_dirs

# Prompt format recommended by the Hunyuan-MT model card for XX -> ZH.
PROMPT_ZH = "把下面的文本翻译成{lang}，不要额外解释。\n\n{text}"
PROMPT_OTHER = "Translate the following segment into {lang}, without additional explanation.\n\n{text}"


# For interim text: the default prompt makes the model "finish" half sentences, invent content and
# add bracketed notes; this wording keeps it to what was actually said.
PROMPT_ZH_FRAGMENT = "把下面这段还没说完的话翻译成{lang}，只翻译已有的内容，不要补全，不要加括号说明。\n\n{text}"
PROMPT_OTHER_FRAGMENT = ("Translate this unfinished sentence into {lang}. Translate only what is there, "
                         "do not complete it, no notes.\n\n{text}")
# Hunyuan-MT adds bracketed notes when the source is ambiguous or misheard: the original English
# after a transliteration, an alternative reading "（或者……）", an explanation. Useful for documents,
# noise for subtitles — and asking for no notes in the prompt does not stop it, so strip them.
_NOTES = re.compile(r"\s*[（(【\[][^（()）【\[\]】]*[）)】\]]")


def strip_notes(text: str) -> str:
    prev = None
    while prev != text:  # repeat for nested brackets
        prev, text = text, _NOTES.sub("", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def build_prompt(text: str, target: str, fragment: bool = False) -> str:
    zh = target in ("中文", "简体中文", "繁体中文", "粤语")
    if fragment:
        tpl = PROMPT_ZH_FRAGMENT if zh else PROMPT_OTHER_FRAGMENT
    else:
        tpl = PROMPT_ZH if zh else PROMPT_OTHER
    return tpl.format(lang=target, text=text)


class Translator:
    def __init__(self, base_url: str, model: str, api_key: str = "", target: str = "中文"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.target = target
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(headers=headers, timeout=httpx.Timeout(30.0, connect=5.0))

    def translate(self, text: str, fragment: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_prompt(text, self.target, fragment)}],
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
        return strip_notes(r.json()["choices"][0]["message"]["content"])

    def close(self) -> None:
        self.client.close()


LOCAL_MODEL_URL = "https://huggingface.co/mradermacher/Hunyuan-MT-7B-GGUF/resolve/main/{file}"


def ensure_local_model(model_file: str, progress) -> None:
    """Download the GGUF on first use (resumable). progress(done_bytes, total_bytes)."""
    path = MODELS_DIR / model_file
    if path.exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    url = LOCAL_MODEL_URL.format(file=model_file)
    with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                      timeout=httpx.Timeout(60.0, connect=15.0)) as r:
        if r.status_code == 200:
            done = 0  # server ignored the range, start over
        elif r.status_code != 206:
            raise RuntimeError(f"下载翻译模型失败：HTTP {r.status_code}")
        total = done + int(r.headers.get("Content-Length", 0))
        last = 0.0
        with open(part, "ab" if done else "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 0.5:
                    progress(done, total)
                    last = time.time()
    part.replace(path)


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
        # the release zip ships cuBLAS only once (for Whisper); let llama-server find it too
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(LLAMA_DIR)] + [str(d) for d in cuda_dll_dirs()] + [env.get("PATH", "")])
        self.proc = subprocess.Popen(
            cmd, cwd=str(LLAMA_DIR), env=env,
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
