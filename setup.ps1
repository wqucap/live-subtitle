# One-time setup from source: Python venv + llama.cpp (CUDA) + Hunyuan-MT-7B model.
# The Whisper model downloads automatically on first start.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot

$llamaTag = "b11487"
$modelUrl = "https://huggingface.co/mradermacher/Hunyuan-MT-7B-GGUF/resolve/main/Hunyuan-MT-7B.Q4_K_M.gguf"

if (-not (Test-Path .venv)) {
    # Use python.org Python; Anaconda's bundled VC++ runtime is too old for PySide6 6.11
    python -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

New-Item -ItemType Directory -Force bin, models | Out-Null
if (-not (Test-Path bin\llama\llama-server.exe)) {
    $base = "https://github.com/ggml-org/llama.cpp/releases/download/$llamaTag"
    foreach ($z in "llama-$llamaTag-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip") {
        Write-Host "Downloading $z"
        Invoke-WebRequest "$base/$z" -OutFile "$env:TEMP\$z"
        Expand-Archive "$env:TEMP\$z" -DestinationPath bin\llama -Force
        Remove-Item "$env:TEMP\$z"
    }
    # keep only what llama-server needs; cuBLAS comes from the nvidia-cublas-cu12 pip package (shared with Whisper)
    Get-ChildItem bin\llama -File | Where-Object {
        -not ($_.Name -notlike 'cublas*' -and ($_.Name -eq 'llama-server.exe' -or $_.Name -like 'LICENSE*' -or
              ($_.Extension -eq '.dll' -and ($_.Name -notlike '*-impl.dll' -or $_.Name -eq 'llama-server-impl.dll'))))
    } | ForEach-Object { Remove-Item -LiteralPath $_.FullName }
}
if (-not (Test-Path models\Hunyuan-MT-7B.Q4_K_M.gguf)) {
    Write-Host "Downloading Hunyuan-MT-7B (about 4.3 GB)"
    curl.exe -L -C - -o models\Hunyuan-MT-7B.Q4_K_M.gguf $modelUrl
}
Write-Host "Setup done. Run: .\.venv\Scripts\python.exe app\main.py   or build the exe with .\build.ps1"
