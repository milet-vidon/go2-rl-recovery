param(
    [ValidateSet("recovery", "locomotion", "standard", "standard_stance")]
    [string]$Stage = "recovery",
    [int]$NumEnvs = 128,
    [int]$MaxIterations = 0,
    [string]$LoadRun = "",
    [string]$Checkpoint = "model_.*.pt",
    [string]$RunName = "",
    [string]$Device = "cuda:0",
    [double]$FallAngleDeg = 0,
    [switch]$FocusSide,
    [switch]$Headless,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$IsaacRoot = "E:\IsaacLab"
$EnvRoot = Join-Path $IsaacRoot "env"
$RepoRoot = Join-Path $IsaacRoot "repo"

# Keep every Isaac Sim user directory, cache, and temporary file on E:.
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
if ($FallAngleDeg -gt 0) {
    $env:ISAACLAB_RECOVERY_FALL_ANGLE_DEG = "$FallAngleDeg"
} else {
    Remove-Item Env:ISAACLAB_RECOVERY_FALL_ANGLE_DEG -ErrorAction SilentlyContinue
}
if ($FocusSide) { $env:ISAACLAB_RECOVERY_FOCUS_SIDE = "1" } else { Remove-Item Env:ISAACLAB_RECOVERY_FOCUS_SIDE -ErrorAction SilentlyContinue }

if ($Stage -eq "locomotion" -and [string]::IsNullOrWhiteSpace($LoadRun)) {
    throw "Locomotion fine-tuning requires -LoadRun <recovery run folder> so it starts from a recovery checkpoint."
}

$Task = if ($Stage -eq "recovery") {
    "Isaac-Recovery-Flat-Unitree-Go2-v0"
} elseif ($Stage -eq "standard_stance") {
    "Isaac-Standard-Flat-Unitree-Go2-v0"
} elseif ($Stage -eq "standard" -or $Stage -eq "standard_stance") {
    "Isaac-Velocity-Flat-Unitree-Go2-v0"
} else {
    "Isaac-Recovery-Locomotion-Flat-Unitree-Go2-v0"
}
if ($MaxIterations -le 0) {
    $MaxIterations = if ($Stage -eq "recovery") { 3000 } elseif ($Stage -eq "standard" -or $Stage -eq "standard_stance") { 4000 } else { 4000 }
}
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = $Stage
}

Push-Location $RepoRoot
try {
    $TrainingArgs = @(
        "-p", "scripts\reinforcement_learning\rsl_rl\train.py",
        "--task", $Task,
        "--num_envs", "$NumEnvs",
        "--max_iterations", "$MaxIterations",
        "--run_name", $RunName,
        "--device", $Device,
        "--rendering_mode", "performance",
        "--kit_args=--/app/vulkan=false"
    )
    if (-not [string]::IsNullOrWhiteSpace($LoadRun)) {
        # Both stages intentionally use the same experiment root. RSL-RL resolves
        # this checkpoint before it creates the new locomotion log directory.
        $TrainingArgs += @("--resume", "--load_run", $LoadRun, "--checkpoint", $Checkpoint)
    }
    if ($Headless) {
        $TrainingArgs += "--headless"
    } else {
        $TrainingArgs += @(
            "env.viewer.eye=[3.0,3.0,1.8]",
            "env.viewer.lookat=[0.0,0.0,0.3]"
        )
    }

    Write-Host "Starting $Stage training: task=$Task, envs=$NumEnvs, iterations=$MaxIterations, device=$Device"
    if ($DryRun) {
        Write-Host ("isaaclab.bat " + ($TrainingArgs -join " "))
        return
    }
    & ".\isaaclab.bat" @TrainingArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
