param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Task = 'Isaac-Recovery-Stable-Flat-Unitree-Go2-Play-v0',
    [int]$Trials = 20,
    [double]$AngleDeg = 30,
    [ValidateRange(0, 60)][double]$SettleSeconds = 0,
    [string]$StateBankPath = '',
    [ValidateSet('heldout', 'train')][string]$StateBankSplit = 'heldout',
    [int]$Seed = 20260909,
    [string[]]$Poses = @('side', 'fore_aft'),
    [ValidateSet('oblique', 'front', 'side')][string]$View = 'oblique',
    [switch]$Video
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw "Missing checkpoint: $Checkpoint" }
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($OutputDir)) -ne 'E:\') { throw 'Outputs must be on E:.' }
if (-not [string]::IsNullOrWhiteSpace($StateBankPath)) {
    if ($Task -ne 'Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0') {
        throw 'StateBankPath requires Task Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0.'
    }
    if ($SettleSeconds -lt 1) { throw 'State-bank replay requires SettleSeconds >= 1 for nominal-PD handover.' }
    if (-not (Test-Path -LiteralPath $StateBankPath)) { throw "Missing state bank: $StateBankPath" }
    if (-not $PSBoundParameters.ContainsKey('Poses')) { $Poses = @('side', 'upside_down') }
    if (@($Poses | Where-Object { $_ -notin @('side', 'upside_down') }).Count -gt 0) {
        throw 'State-bank poses must be side (actual left/right) or upside_down (actual back).'
    }
}
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
    '--angle_deg', "$AngleDeg", '--settle_s', "$SettleSeconds", '--seed', "$Seed", '--horizon_s', '8', '--hold_s', '3',
    '--min_contacts', '4', '--device', 'cuda:0', '--headless', '--kit_args=--/app/vulkan=false', '--poses') + $Poses
if ($Video) { $evalArgs += @('--video_pose', 'all', '--view', $View) }
if (-not [string]::IsNullOrWhiteSpace($StateBankPath)) {
    $evalArgs += @('--state_bank_path', $StateBankPath, '--state_bank_split', $StateBankSplit)
}
$started = Get-Date
$previousBankCollection = [Environment]::GetEnvironmentVariable('ISAACLAB_RECOVERY_BANK_COLLECTION', 'Process')
try {
    if (-not [string]::IsNullOrWhiteSpace($StateBankPath)) { $env:ISAACLAB_RECOVERY_BANK_COLLECTION = '1' }
    & 'E:\IsaacLab\env\python.exe' @evalArgs
    if ($LASTEXITCODE -ne 0) { throw "Recovery evaluation exited with $LASTEXITCODE" }
}
finally {
    if (-not [string]::IsNullOrWhiteSpace($StateBankPath)) {
        if ($null -eq $previousBankCollection) {
            Remove-Item Env:ISAACLAB_RECOVERY_BANK_COLLECTION -ErrorAction SilentlyContinue
        } else {
            $env:ISAACLAB_RECOVERY_BANK_COLLECTION = $previousBankCollection
        }
    }
}
$report = Join-Path $OutputDir ((Split-Path $Checkpoint -Leaf) + '_recovery_metrics.json')
if (-not (Test-Path -LiteralPath $report) -or (Get-Item -LiteralPath $report).LastWriteTime -lt $started) {
    throw 'Simulator did not produce a fresh report.'
}
