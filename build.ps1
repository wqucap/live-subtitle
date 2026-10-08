# Builds LiveSubtitle.exe + _internal\ into this folder, next to bin\ and models\.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

& $py make_icon.py
$app = Join-Path $PSScriptRoot "app"
& $py -m PyInstaller "$app\main.py" --noconfirm --clean --onedir --windowed `
    --name LiveSubtitle --icon "$app\icon.ico" --add-data "$app\icon.ico;." `
    --paths $app --distpath build\dist --workpath build\work --specpath build `
    --collect-data faster_whisper --collect-binaries ctranslate2 `
    --collect-binaries nvidia.cublas --collect-binaries nvidia.cudnn `
    --exclude-module tkinter --exclude-module matplotlib
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

if (Get-Process LiveSubtitle -ErrorAction SilentlyContinue) {
    Write-Host "LiveSubtitle.exe is running - new build left in build\dist\LiveSubtitle (close the app and re-run to install it here)"
    exit 0
}
if (Test-Path _internal) { Remove-Item -Recurse -Force _internal }
Move-Item build\dist\LiveSubtitle\_internal .
Move-Item -Force build\dist\LiveSubtitle\LiveSubtitle.exe .
Remove-Item -Recurse -Force build  # intermediate files only; keeps the folder free of a second copy
Write-Host "Done: $PSScriptRoot\LiveSubtitle.exe"
