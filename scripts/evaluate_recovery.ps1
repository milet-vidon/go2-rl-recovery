param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Task = 'Isaac-Recovery-Stable-Flat-Unitree-Go2-Play-v0',
    [int]$Trials = 20,
    [double]$AngleDeg = 30,
    [int]$Seed = 20260909,
    [string[]]$Poses = @('side', 'fore_aft'),
    [ValidateSet('oblique', 'front', 'side')][string]$View = 'oblique',
    [switch]$Video
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw "Missing checkpoint: $Checkpoint" }
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($OutputDir)) -ne 'E:\') { throw 'Outputs must be on E:.' }
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:OMNI_USER_HOME = 'E:\IsaacLab\userdata'
$env:OV_USER_HOME = 'E:\IsaacLab\userdata'
$env:TEMP = 'E:\IsaacLab\tmp'
$env:TMP = 'E:\IsaacLab\tmp'
$env:PIP_CACHE_DIR = 'E:\IsaacLab\cache\pip'
$env:PYTHONIOENCODING = 'utf-8'
$env:ISAACLAB_GO2_USD = 'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd'
$env:ISAACLAB_GROUND_USD = 'E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd'
$env:CONDA_PREFIX = 'E:\IsaacLab\env'
$env:Path = 'E:\IsaacLab\env;E:\IsaacLab\env\Scripts;' + $env:Path
$evalArgs = @((Join-Path $PSScriptRoot 'evaluate_go2_recovery.py'), '--task', $Task,
    '--checkpoint', $Checkpoint, '--output_dir', $OutputDir, '--trials', "$Trials",
    '--angle_deg', "$AngleDeg", '--seed', "$Seed", '--horizon_s', '8', '--hold_s', '3',
    '--min_contacts', '4', '--device', 'cuda:0', '--headless', '--kit_args=--/app/vulkan=false', '--poses') + $Poses
if ($Video) { $evalArgs += @('--video_pose', 'all', '--view', $View) }
$started = Get-Date
& 'E:\IsaacLab\env\python.exe' @evalArgs
if ($LASTEXITCODE -ne 0) { throw "Recovery evaluation exited with $LASTEXITCODE" }
$report = Join-Path $OutputDir ((Split-Path $Checkpoint -Leaf) + '_recovery_metrics.json')
if (-not (Test-Path -LiteralPath $report) -or (Get-Item -LiteralPath $report).LastWriteTime -lt $started) {
    throw 'Simulator did not produce a fresh report.'
}
