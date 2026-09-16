param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag,
    [ValidateRange(100,3000)][int]$Iterations = 2000,
    [switch]$PreflightOnly
)
# Finite post-training recording, NOT an integrated recovery/locomotion controller.
$ErrorActionPreference = 'Stop'
$portfolio = Split-Path $PSScriptRoot -Parent
$checkpointName = "model_$($Iterations - 1).pt"
$output = Join-Path $portfolio "evaluations/$RunTag-video-review"
$logRoot = 'E:\IsaacLab\artifacts\recovery-20260916'
$bankPath = Join-Path $portfolio 'datasets/recovery_states/nominal_pd_v1_20260916'
$locomotion = Join-Path $portfolio 'models/locomotion/natural_stop_diagonal_model2250.pt'
$python = 'E:\IsaacLab\env\python.exe'
$reports = @{}
$pending = @()
foreach ($arm in @('nominal','current')) {
    foreach ($protocol in @('heldout','angle30','angle45')) {
        $path = Join-Path $portfolio "evaluations/$RunTag-$arm-$protocol/$($checkpointName)_recovery_metrics.json"
        if (-not (Test-Path -LiteralPath $path)) { $pending += $path; continue }
        $reports["$arm-$protocol"] = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    }
}
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'E:[/\\]IsaacLab' -and
    $_.CommandLine -match 'rsl_rl[/\\]train.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py'
})
if ($pending.Count -or $active.Count) {
    Write-Output "NOT READY: $($pending.Count) missing completed reports; active simulator PIDs: $($active.ProcessId -join ','). No recording started."
    if ($PreflightOnly) { return }
    throw 'Wait for the existing serial pair and all evaluations. Do not run a second simulator.'
}
function Assert-HealthyLog([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing simulation log: $Path" }
    $bad = Select-String -LiteralPath $Path -Pattern 'CUDA error|PhysX error|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($null -ne $bad) { throw "Invalid simulation: $Path : $($bad.Line)" }
}
$snapshot = Get-Content -LiteralPath (Join-Path $portfolio "configs/$RunTag-source-snapshot.json") -Raw | ConvertFrom-Json
foreach ($file in $snapshot.files) {
    if ((Get-FileHash -LiteralPath $file.path -Algorithm SHA256).Hash -ne $file.sha256) { throw "Training source drift: $($file.path)" }
}
foreach ($arm in @('nominal','current')) {
    $variant = if ($arm -eq 'nominal') { 'NominalTarget' } else { 'CurrentTarget' }
    $task = "Isaac-Recovery-Bank-$variant-Flat-Unitree-Go2-Play-v0"
    $model = $reports["$arm-heldout"]
    $sha = (Get-FileHash -LiteralPath $model.checkpoint -Algorithm SHA256).Hash
    foreach ($protocol in @('heldout','angle30','angle45')) {
        $r = $reports["$arm-$protocol"]
        if ($r.task -ne $task -or $r.checkpoint_sha256 -ne $sha -or $r.acceptance_eligible -ne $true -or $r.policy_action_mode -ne 'deterministic_mean') {
            throw "Invalid source evaluation for $arm $protocol"
        }
        Assert-HealthyLog (Join-Path $logRoot "$RunTag-$arm-$protocol.log")
    }
    if ($model.state_bank.sha256 -ne (Get-FileHash -LiteralPath (Join-Path $bankPath 'states.npz') -Algorithm SHA256).Hash -or $model.state_bank.split -ne 'heldout') {
        throw 'Source bank or split mismatch'
    }
    Assert-HealthyLog (Join-Path $logRoot "$RunTag-$arm-train.log")
}
if (-not (Test-Path -LiteralPath $locomotion)) { throw 'Missing retained locomotion baseline' }
if (Test-Path -LiteralPath $output) { throw "Existing video output: $output; inspect/resume missing stages manually, never overwrite." }
if ($PreflightOnly) { Write-Output 'READY: all source reports and physics logs valid; no recording started.'; return }

New-Item -ItemType Directory -Path $output | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = 'E:\IsaacLab\tmp'; $env:TMP = 'E:\IsaacLab\tmp'
$env:PYTHONIOENCODING = 'utf-8'
Write-Output 'Separate-policy scenario review. Baseline2250 is NOT either new recovery actor. All failures retained.'
foreach ($case in @('normal','push')) {
    foreach ($view in @('front','oblique')) {
        $label = "retained_locomotion2250_$($case)_$view"
        $dir = Join-Path $output $label
        $log = Join-Path $output "$label.log"
        $push = if ($case -eq 'push') { 0.75 } else { 0.0 }
        & (Join-Path $PSScriptRoot 'evaluate_stand_walk_stop.ps1') -Checkpoint $locomotion -OutputDir $dir -Seed 20260918 -View $view -PushSpeed $push *> $log
        if ($LASTEXITCODE -ne 0) { throw "Recording failed: $log" }
        Assert-HealthyLog $log
        $report = Join-Path $dir 'natural_stop_diagonal_model2250_stand_walk_stop.json'
        & $python (Join-Path $PSScriptRoot 'build_validation_media.py') $report (Join-Path $output 'locomotion-media') --label $label *> (Join-Path $output "$label-encode.log")
        if ($LASTEXITCODE -ne 0) { throw "Encoding failed: $label" }
        Write-Output "Recorded $label; inspect the complete rollout and its actual passed flag."
    }
}
foreach ($arm in @('nominal','current')) {
    $model = $reports["$arm-heldout"]
    foreach ($scenario in @('heldout','upright')) {
        foreach ($view in @('front','oblique')) {
            $label = "$arm-$scenario-$view"
            $dir = Join-Path $output $label
            $log = Join-Path $output "$label.log"
            $parameters = @{Checkpoint=$model.checkpoint;Task=$model.task;OutputDir=$dir;Trials=1;Seed=20260918;View=$view;Video=$true}
            if ($scenario -eq 'heldout') {
                $parameters.StateBankPath=$bankPath; $parameters.StateBankSplit='heldout'; $parameters.SettleSeconds=1
                $parameters.Poses=@('side','upside_down')
            } else { $parameters.Poses=@('upright'); $parameters.AngleDeg=30 }
            & (Join-Path $PSScriptRoot 'evaluate_recovery.ps1') @parameters *> $log
            if ($LASTEXITCODE -ne 0) { throw "Recording failed: $log" }
            Assert-HealthyLog $log
            Write-Output "Recorded $label; this is a diagnostic, not a success declaration."
        }
        $front = Join-Path $output "$arm-$scenario-front/$($checkpointName)_recovery_metrics.json"
        $oblique = Join-Path $output "$arm-$scenario-oblique/$($checkpointName)_recovery_metrics.json"
        & $python (Join-Path $PSScriptRoot 'pair_recovery_views.py') $front $oblique (Join-Path $output "$arm-$scenario-paired") *> (Join-Path $output "$arm-$scenario-pair.log")
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Views not paired for $arm ${scenario}: retain both raw reports/videos and inspect the mismatch. Do not claim synchronized views."
        }
    }
}
Write-Output "Recording finished at $output. Visual inspection and honest result delivery are still REQUIRED before starting another long training branch."
