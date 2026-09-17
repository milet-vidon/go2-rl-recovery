param(
    [Parameter(Mandatory=$true)][ValidateSet('A','B')][string]$Arm,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag,
    [ValidateSet(2,50)][int]$Iterations = 2,
    [ValidateSet(16,128)][int]$NumEnvs = 16,
    [switch]$PreflightOnly
)
# ONE independently resumed arm; default is smoke. No second block, eval,
# sibling arm, auto-extension, acceptance, or modification of old artifacts.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not (($NumEnvs -eq 16 -and $Iterations -eq 2) -or ($NumEnvs -eq 128 -and $Iterations -eq 50))) {
    throw 'Only 16 envs/2 updates smoke or one 128 envs/50 updates block is allowed.'
}
$portfolio = Split-Path $PSScriptRoot -Parent
$repo = 'E:\IsaacLab\repo'
$python = 'E:\IsaacLab\env\python.exe'
$runRoot = Join-Path $repo 'logs/rsl_rl/unitree_go2_flat'
$parentRun = '2026-09-17_03-10-21_20260917-speedretention128x300'
$parent = Join-Path $runRoot "$parentRun/model_3947.pt"
$logRoot = Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
$config = Join-Path $portfolio "configs/$RunTag"
$guard = Join-Path $PSScriptRoot 'check_speed_tracking_weight_block.py'
$preservationScript = Join-Path $PSScriptRoot 'verify_overnight_baselines.py'
$preservationManifest = Join-Path $portfolio 'configs/overnight_preservation_20260917.json'
$trainLog = Join-Path $logRoot 'train.log'
$weight = if ($Arm -eq 'A') { '1.5' } else { '2.0' }
$expectedFinal = "model_$(3947 + $Iterations - 1).pt"
$expectedAdam = 79120 + $Iterations * 20

function Assert-Idle {
    # Conservative and never kills: future simulator filenames and CPU validators
    # in this environment also block. Do not overlap with a recovery pipeline.
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit|isaac-sim.*)\.exe$' -and
        ($_.CommandLine -match 'E:[/\\]IsaacLab(?:[/\\]|\b)' -or
         $_.ExecutablePath -match '^E:[/\\]IsaacLab(?:[/\\]|\b)')
    })
    if ($active.Count) { throw "Workspace Python/Kit active: $($active.ProcessId -join ','). Wait; do not overlap." }
}
function Assert-NewOutputs {
    foreach ($path in @($logRoot,$config)) {
        if (Test-Path -LiteralPath $path) { throw "Existing output: $path. Do not overwrite or append." }
    }
    if (@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") }).Count) {
        throw 'Existing training run tag.'
    }
    $matches = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name -match $parentRun })
    if ($matches.Count -ne 1 -or $matches[0].Name -ne $parentRun) { throw 'Ambiguous official trainer load_run regex.' }
}
function Assert-Sources {
    foreach ($source in $script:sources) {
        if ((Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash -ne $source.sha256) {
            throw "Frozen parent/source drift: $($source.path)"
        }
    }
}
function Assert-HealthyLog([string]$Path) {
    $bad = Select-String -LiteralPath $Path -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|\[Error\].*PhysX|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($null -ne $bad) { throw "Invalid training log: $Path : $($bad.Line)" }
}
function Assert-Preserved([string]$Phase,[string]$LogPath = '') {
    # Pin the checker BEFORE execution. -I ignores PYTHONOPTIMIZE so its legacy
    # assert statements cannot silently disappear; -B keeps this read-only.
    if ((Get-FileHash -LiteralPath $preservationScript -Algorithm SHA256).Hash -ne '2FBD670D819278CAA19142D77BC65ECC74F960E4AFC19E8142FEE3E507CBA41C' -or
        (Get-FileHash -LiteralPath $preservationManifest -Algorithm SHA256).Hash -ne '0250A8C60DB31FB21DF03FFD16591D8C8E9BFE61E1C0D85A76E91C2929648E55') {
        throw 'Frozen preservation checker/manifest changed.'
    }
    $preservationResult = & $python -I -B $preservationScript 2>&1
    $preservationExit = $LASTEXITCODE
    if ($LogPath) { $preservationResult | Set-Content -LiteralPath $LogPath -Encoding UTF8 }
    if ($preservationExit -ne 0) { throw "16-model preservation failed ($Phase): $($preservationResult -join ' ')" }
    Write-Output "Preservation $Phase passed: all 16 exported old models unchanged (not behavioral acceptance)."
}

Assert-Idle
foreach ($path in @($portfolio,$repo,$python,$runRoot,$parent,$logRoot,$config,$guard)) {
    if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($path)) -ne 'E:\') { throw "Non-E-drive path: $path" }
}
Assert-NewOutputs
# The read-only guard uses a restricted small pickle/float reader, never torch.
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = 'E:\IsaacLab\tmp'
$env:TMP = 'E:\IsaacLab\tmp'
Assert-Preserved 'before-preflight'
$guardArgs = @('--arm',$Arm,'--num-envs',"$NumEnvs",'--iterations',"$Iterations",'--run-name',$RunTag)
try {
    $preflightText = & $python -B $guard @guardArgs --preflight-only
    if ($LASTEXITCODE -ne 0) { throw 'Frozen source/config/actual checkpoint preflight failed.' }
}
finally { Assert-Preserved 'after-preflight' }
$preflight = ($preflightText -join [Environment]::NewLine) | ConvertFrom-Json
if ($preflight.status -ne 'passed' -or $preflight.quality_accepted -ne $false) { throw 'Unexpected preflight result.' }
$script:sources = @($preflight.sources)
Assert-Sources
Assert-Idle
Write-Output "ONE weight arm $Arm ($weight), same frozen control3947; $NumEnvs envs/$Iterations updates; endpoint $expectedFinal, Adam $expectedAdam."
Write-Output 'Same vx[-0.5,1.0], 30% stand, reward kernel/PPO/physics/pushes. Full actor/Adam/std resume; LR scalar=actual saved1e-5; not exact RNG/simulator continuity.'
if ($PreflightOnly) { Write-Output 'Preflight passed; no output created and no simulator started.'; return }

Assert-NewOutputs
New-Item -ItemType Directory -Path $logRoot | Out-Null
$sources | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $logRoot 'source-snapshot.json') -Encoding UTF8
$preflightText | Set-Content -LiteralPath (Join-Path $logRoot 'preflight.json') -Encoding UTF8
$argsTrain = @('-p','scripts\reinforcement_learning\rsl_rl\train.py',
    '--task','Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0','--num_envs',"$NumEnvs",
    '--max_iterations',"$Iterations",'--run_name',$RunTag,'--device','cuda:0',
    '--rendering_mode','performance','--kit_args=--/app/vulkan=false','--headless',
    '--resume','--load_run',$parentRun,'--checkpoint','model_3947.pt',
    "env.rewards.track_lin_vel_xy_exp.weight=$weight",'agent.algorithm.learning_rate=1e-5')
@{ arm=$Arm; weight=$weight; parent=$parent; num_envs=$NumEnvs; updates=$Iterations;
   expected_final=$expectedFinal; expected_adam_step=$expectedAdam; arguments=$argsTrain;
   quality_accepted=$false; training_only=$true; note='One finite block; manual regression required; no second block.'
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $logRoot 'invocation.json') -Encoding UTF8
Push-Location $portfolio
try {
    Assert-Idle
    Assert-Sources
    # This existing entry sets E: runtime/cache paths, but DryRun never launches.
    & ./scripts/train.ps1 -Stage natural_robust_push -NumEnvs $NumEnvs -MaxIterations $Iterations -LoadRun $parentRun -Checkpoint model_3947.pt -RunName $RunTag -Headless -DryRun
    Assert-Idle
    Assert-Sources
    Assert-Preserved 'before-batch' (Join-Path $logRoot 'preservation-before-batch.log')
    Push-Location $repo
    try {
        & .\isaaclab.bat @argsTrain *> $trainLog
        $trainExit = $LASTEXITCODE
    }
    finally {
        Pop-Location
        # Also execute on a nonzero/throwing training exit; preserve its evidence.
        Assert-Preserved 'after-training' (Join-Path $logRoot 'preservation-after-training.log')
    }
    if ($trainExit -ne 0) { throw "Finite arm training failed: $trainLog" }
    Assert-HealthyLog $trainLog
    Assert-Idle
    Assert-Sources
    $runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") })
    if ($runs.Count -ne 1) { throw 'Expected exactly one new arm run.' }
    $checkpoint = Join-Path $runs[0].FullName $expectedFinal
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Missing endpoint: $checkpoint" }
    New-Item -ItemType Directory -Path $config | Out-Null
    foreach ($name in @('env.yaml','agent.yaml')) {
        Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$name") -Destination (Join-Path $config $name)
    }
    $checkLog = Join-Path $logRoot 'config-checkpoint-check.json'
    & $python -B $guard @guardArgs --candidate-dir $config --checkpoint $checkpoint *> $checkLog
    if ($LASTEXITCODE -ne 0) { throw "Saved config/actual checkpoint guard failed: $checkLog" }
    Assert-Sources
    Assert-Preserved 'after-batch-validation' (Join-Path $logRoot 'preservation-after-validation.log')
    Write-Output "Finite TRAINING ONLY complete: $checkpoint; SHA256 $((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash)."
    Write-Output "Configuration/68 finite tensors/actual Adam counters verified: $checkLog"
    Write-Output 'NOT ACCEPTED. No auto evaluation, promotion, sibling arm or extension; manual regression decision required.'
}
finally { Pop-Location }
