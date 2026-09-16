param(
    [string]$Checkpoint = "E:\IsaacLab\go2-rl-open-source\models\locomotion\natural_stop_model1349.pt",
    [string]$Task = "Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0",
    [string]$OutputDir = "E:\IsaacLab\artifacts\codex-2026-09-06-new-chat\outputs\stand_walk_stop\manual",
    [double]$PushSpeed = 0,
    [double]$WalkSpeed = 0.5,
    [double]$LateralSpeed = 0,
    [double]$YawRate = 0,
    [int]$Seed = 20260909,
    [ValidateSet('legacy','front','oblique')][string]$View = 'legacy',
    [switch]$ZeroAction,
    [switch]$NoVideo
)
$ErrorActionPreference = "Stop"
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($OutputDir)) -ne 'E:\') { throw 'Outputs must be on E:.' }
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:OMNI_USER_HOME = "E:\IsaacLab\userdata"
$env:OV_USER_HOME = "E:\IsaacLab\userdata"
$env:TEMP = "E:\IsaacLab\tmp"
$env:TMP = "E:\IsaacLab\tmp"
$env:PIP_CACHE_DIR = "E:\IsaacLab\cache\pip"
$env:PYTHONIOENCODING = "utf-8"
$env:ISAACLAB_GO2_USD = "E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd"
$env:ISAACLAB_GROUND_USD = "E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd"
$env:CONDA_PREFIX = "E:\IsaacLab\env"
$env:Path = "E:\IsaacLab\env;E:\IsaacLab\env\Scripts;" + $env:Path
$evalArgs = @((Join-Path $PSScriptRoot 'evaluate_go2_stand_walk_stop.py'), "--task", $Task,
    "--output_dir", $OutputDir, "--seed", "$Seed", "--push_speed", "$PushSpeed",
    "--walk_speed", "$WalkSpeed", "--lateral_speed", "$LateralSpeed", "--yaw_rate", "$YawRate",
    "--view", $View, "--device", "cuda:0", "--headless", "--rendering_mode", "performance", "--kit_args=--/app/vulkan=false")
if ($ZeroAction) { $evalArgs += "--zero_action" } else { $evalArgs += @("--checkpoint", $Checkpoint) }
if ($NoVideo) { $evalArgs += "--no_video" }
Push-Location "E:\IsaacLab\repo"
$started = Get-Date
try {
    & "E:\IsaacLab\env\python.exe" @evalArgs
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
$stem = if ($ZeroAction) { 'zero_action' } else { [IO.Path]::GetFileNameWithoutExtension($Checkpoint) }
$report = Join-Path $OutputDir ($stem + '_stand_walk_stop.json')
if (-not (Test-Path -LiteralPath $report) -or (Get-Item -LiteralPath $report).LastWriteTime -lt $started) {
    throw 'Simulator did not produce a fresh stand/walk/stop report.'
}
