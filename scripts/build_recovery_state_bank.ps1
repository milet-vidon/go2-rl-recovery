param(
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Task = 'Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0',
    [int]$NumEnvs = 32,
    [int]$Batches = 8,
    [int]$Seed = 20260916,
    [double]$SettleSeconds = 2,
    [double]$MaxSettleSeconds = 5,
    [double]$ReplaySeconds = 1,
    [double]$QuietHoldSeconds = 0.5,
    [int]$HeldoutEvery = 5,
    [string]$Device = 'cuda:0'
)
$ErrorActionPreference = 'Stop'
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
$previousCollectionMode = $env:ISAACLAB_RECOVERY_BANK_COLLECTION
$env:ISAACLAB_RECOVERY_BANK_COLLECTION = '1'
$collectorArgs = @((Join-Path $PSScriptRoot 'build_recovery_state_bank.py'), '--task', $Task,
    '--output_dir', $OutputDir, '--num_envs', "$NumEnvs", '--batches', "$Batches", '--seed', "$Seed",
    '--settle_s', "$SettleSeconds", '--max_settle_s', "$MaxSettleSeconds", '--replay_s', "$ReplaySeconds",
    '--quiet_hold_s', "$QuietHoldSeconds", '--heldout_every', "$HeldoutEvery", '--device', $Device,
    '--headless', '--kit_args=--/app/vulkan=false')
$started = Get-Date
try {
    & 'E:\IsaacLab\env\python.exe' @collectorArgs
    if ($LASTEXITCODE -ne 0) { throw "State-bank collection exited with $LASTEXITCODE" }
    $manifestPath = Join-Path $OutputDir 'manifest.json'
    $statesPath = Join-Path $OutputDir 'states.npz'
    if (-not (Test-Path -LiteralPath $manifestPath) -or -not (Test-Path -LiteralPath $statesPath)) {
        throw 'Collector did not produce the expected states.npz and manifest.json.'
    }
    if ((Get-Item -LiteralPath $manifestPath).LastWriteTime -lt $started) { throw 'Manifest is not fresh.' }
    $bankManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($bankManifest.status -ne 'validated' -or $bankManifest.replay_validation.statistics.accepted_states -lt 1) {
        throw 'No validated states were produced.'
    }
    Write-Host "Validated state bank: $statesPath"
} finally {
    if ($null -eq $previousCollectionMode) {
        Remove-Item Env:ISAACLAB_RECOVERY_BANK_COLLECTION -ErrorAction SilentlyContinue
    } else {
        $env:ISAACLAB_RECOVERY_BANK_COLLECTION = $previousCollectionMode
    }
}
