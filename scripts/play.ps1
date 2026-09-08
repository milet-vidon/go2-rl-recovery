param(
    [ValidateSet("recovery", "locomotion", "standard", "standard_stance")]
    [string]$Stage = "locomotion",
    [string]$Checkpoint = "",
    [int]$VideoLength = 600,
    [switch]$Video,
    [switch]$ForwardCommand,
    [switch]$Static,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$IsaacRoot = "E:\IsaacLab"
$EnvRoot = Join-Path $IsaacRoot "env"
$RepoRoot = Join-Path $IsaacRoot "repo"

$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:OMNI_USER_HOME = Join-Path $IsaacRoot "userdata"
$env:OV_USER_HOME = Join-Path $IsaacRoot "userdata"
$env:TEMP = Join-Path $IsaacRoot "tmp"
$env:TMP = Join-Path $IsaacRoot "tmp"
$env:PIP_CACHE_DIR = Join-Path $IsaacRoot "cache\pip"
$env:PYTHONIOENCODING = "utf-8"
$localGo2Usd = Join-Path $IsaacRoot "userdata\assets\Robots\Unitree\Go2\go2.usd"
if (Test-Path -LiteralPath $localGo2Usd) {
    $env:ISAACLAB_GO2_USD = $localGo2Usd
}
$localGroundUsd = Join-Path $IsaacRoot "userdata\assets\Environments\Grid\default_environment.usd"
if (Test-Path -LiteralPath $localGroundUsd) {
    $env:ISAACLAB_GROUND_USD = $localGroundUsd
}
$env:CONDA_PREFIX = $EnvRoot
$env:CONDA_DEFAULT_ENV = "isaaclab"
$env:Path = "$EnvRoot;$EnvRoot\Scripts;$env:Path"

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    throw "Pass -Checkpoint with the full E: path to model_*.pt."
}
$Task = if ($Stage -eq "recovery") {
    "Isaac-Recovery-Flat-Unitree-Go2-Play-v0"
} elseif ($Stage -eq "standard_stance") {
    "Isaac-Standard-Flat-Unitree-Go2-Play-v0"
} elseif ($Stage -eq "standard") {
    "Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
} else {
    "Isaac-Recovery-Locomotion-Flat-Unitree-Go2-Play-v0"
}

Push-Location $RepoRoot
try {
    $PlayArgs = @(
        "-p", "scripts\reinforcement_learning\rsl_rl\play.py",
        "--task", $Task,
        "--num_envs", "1",
        "--checkpoint", $Checkpoint,
        "--device", "cuda:0",
        "--real-time",
        "--rendering_mode", "performance",
        "env.viewer.eye=[1.8,1.8,1.0]",
        "env.viewer.lookat=[0.0,0.0,0.3]",
        "--kit_args=--/app/vulkan=false"
    )
    if ($ForwardCommand -and $Stage -eq "locomotion") {
        $PlayArgs += @(
            "env.commands.base_velocity.ranges.lin_vel_x=[0.5,0.5]",
            "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]",
            "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]",
            "env.commands.base_velocity.ranges.heading=[0.0,0.0]"
        )
    }
    if ($Static -and $Stage -eq "locomotion") {
        $PlayArgs += @(
            "env.commands.base_velocity.ranges.lin_vel_x=[0.0,0.0]",
            "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]",
            "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]",
            "env.commands.base_velocity.ranges.heading=[0.0,0.0]"
        )
    }
    if ($Video) {
        $PlayArgs += @("--video", "--video_length", "$VideoLength")
    }
    Write-Host "Playing $Stage policy: task=$Task"
    if ($DryRun) {
        Write-Host ("isaaclab.bat " + ($PlayArgs -join " "))
        return
    }
    & ".\isaaclab.bat" @PlayArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
