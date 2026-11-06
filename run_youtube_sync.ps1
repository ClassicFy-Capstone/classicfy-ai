param(
    [ValidateSet('run','estimate','status','benchmark')][string]$Mode = 'run',
    [int]$Workers = 2,
    [int]$Limit = 0,
    [string[]]$Ids,
    [string]$InputPath,
    [string]$OutputPath,
    [string]$AudioCache,
    [string]$MidiRoot,
    [string]$PythonExe,
    [switch]$SaveClips,
    [switch]$KeepAudio,
    [switch]$IncludeExcluded,
    [switch]$Offline,
    [switch]$RetryFailed
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

# Preserve caller-relative arguments before changing directory.
foreach ($paramName in @('InputPath','OutputPath','AudioCache','MidiRoot')) {
    $paramValue = Get-Variable -Name $paramName -ValueOnly
    if ($paramValue) { Set-Variable -Name $paramName -Value ([IO.Path]::GetFullPath($paramValue)) }
}
if (-not $PythonExe) {
    $pilotPython = Join-Path $PSScriptRoot '..\outputs\atepp_youtube_sync_pilot\.venv\Scripts\python.exe'
    $isolatedPython = Join-Path $PSScriptRoot '.venv-youtube-sync\Scripts\python.exe'
    if (Test-Path -LiteralPath $isolatedPython) { $PythonExe = $isolatedPython }
    elseif (Test-Path -LiteralPath $pilotPython) { $PythonExe = $pilotPython }
    else { $PythonExe = (Get-Command python -ErrorAction Stop).Source }
}
if ($Mode -in @('run','benchmark')) {
    & $PythonExe -c 'import importlib.util,sys; sys.exit(int(any(importlib.util.find_spec(m) is None for m in ["synctoolbox","yt_dlp","pretty_midi","soundfile","imageio_ffmpeg"])))'
    if ($LASTEXITCODE -ne 0) {
        $taskVenv = Join-Path $PSScriptRoot '.venv-youtube-sync'
        & $PythonExe -m venv $taskVenv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python environment.' }
        $PythonExe = Join-Path $taskVenv 'Scripts\python.exe'
        & $PythonExe -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements-youtube-sync.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    }
}
$taskArgs = @((Join-Path $PSScriptRoot 'classicfy-ai\scripts\match_atepp_youtube.py'), $Mode, '--workers', "$Workers")
if ($Limit -gt 0) { $taskArgs += @('--limit', "$Limit") }
if ($Ids) { $taskArgs += '--ids'; $taskArgs += $Ids }
if ($InputPath) { $taskArgs += @('--input', $InputPath) }
if ($OutputPath) { $taskArgs += @('--output', $OutputPath) }
if ($AudioCache) { $taskArgs += @('--audio-cache', $AudioCache) }
if ($MidiRoot) { $taskArgs += @('--midi-root', $MidiRoot) }
if ($SaveClips) { $taskArgs += '--save-clips' }
if ($KeepAudio) { $taskArgs += '--keep-audio' }
if ($IncludeExcluded) { $taskArgs += '--include-excluded' }
if ($Offline) { $taskArgs += '--offline' }
if ($RetryFailed) { $taskArgs += '--retry-failed' }
& $PythonExe @taskArgs
exit $LASTEXITCODE
