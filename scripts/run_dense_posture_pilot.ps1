param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag = '20260917-denseposture128x600',
    [ValidateSet(2,600)][int]$Iterations = 600,
    [ValidateSet(16,128)][int]$NumEnvs = 128,
    [ValidateSet(2,20)][int]$Trials = 20,
    [switch]$PreflightOnly
)
# Finite single-policy experiment. Every evaluation scene gets a fresh simulator.
# Never promotes a checkpoint, modifies acceptance, or extends training itself.
$ErrorActionPreference = 'Stop'
if (-not (($NumEnvs -eq 16 -and $Iterations -eq 2 -and $Trials -eq 2) -or
          ($NumEnvs -eq 128 -and $Iterations -eq 600 -and $Trials -eq 20))) {
    throw 'Use 16 envs/2 updates/2 trials for smoke or 128 envs/600 updates/20 trials for the formal pilot.'
}
$portfolio = Split-Path $PSScriptRoot -Parent
$artifacts = 'E:\IsaacLab\artifacts\recovery-20260916'
$runRoot = 'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery'
$installed = 'E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$parentRun = '2026-09-16_19-02-40_20260916-smithnominal128x2000'
$parentCheckpoint = Join-Path $runRoot "$parentRun/model_1999.pt"
$parentSha = '71E4C7A39FF81836D2EAD1C4533988C696E5BD4A0FC939B8CEF6B4F6DAEC1B1C'
$bank = Join-Path $portfolio 'datasets/recovery_states/nominal_pd_v1_20260916'
$bankSha = '71B882E92035B4C248E5980339C980DBB242DCB1EC711C05252C33249DB48F5A'
$task = 'Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0'
$python = 'E:\IsaacLab\env\python.exe'
$snapshotPath = Join-Path $artifacts "$RunTag-source-snapshot.json"
$config = Join-Path $portfolio "configs/$RunTag"
$trainLog = Join-Path $artifacts "$RunTag-train.log"
$expectedFinal = "model_$(1999 + $Iterations - 1).pt"
$protocols = @('bank-side','bank-back','upright-pd1','angle30-side','angle30-fore')

function Assert-Idle {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pythonw.exe','kit.exe','isaac-sim.exe') -and
        $_.CommandLine -match 'E:[/\\]IsaacLab' -and
        ($_.Name -in @('kit.exe','isaac-sim.exe') -or $_.CommandLine -match 'rsl_rl[/\\]train.py|resume_smith_checkpoint_entry.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py')
    })
    if ($active.Count) { throw "Another simulator is active: $($active.ProcessId -join ',')." }
}
function Assert-HealthyLog([string]$Path) {
    $bad = Select-String -LiteralPath $Path -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|\[Error\].*PhysX|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($bad) { throw "Invalid simulation: $Path : $($bad.Line)" }
}
Assert-Idle
foreach ($path in @($portfolio,$artifacts,$runRoot,$installed,$bank,$python)) {
    if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($path)) -ne 'E:\') { throw "Non-E-drive path: $path" }
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing dependency: $path" }
}
if ((Get-FileHash -LiteralPath $parentCheckpoint -Algorithm SHA256).Hash -ne $parentSha) { throw 'Parent checkpoint hash mismatch.' }
if ((Get-FileHash -LiteralPath (Join-Path $bank 'states.npz') -Algorithm SHA256).Hash -ne $bankSha) { throw 'State-bank hash mismatch.' }
$sources = @()
foreach ($name in @('env.yaml','agent.yaml')) {
    $actual = Join-Path $runRoot "$parentRun/params/$name"
    $archive = Join-Path $portfolio "configs/20260916-smithnominal128x2000/$name"
    if ((Get-FileHash -LiteralPath $actual).Hash -ne (Get-FileHash -LiteralPath $archive).Hash) { throw "Parent config archive mismatch: $name" }
    $sources += @{path=$actual;sha256=(Get-FileHash -LiteralPath $actual).Hash}, @{path=$archive;sha256=(Get-FileHash -LiteralPath $archive).Hash}
}
foreach ($name in @('standard_env_cfg.py','robust_env_cfg.py','natural_env_cfg.py','natural_stop_env_cfg.py',
                   'natural_robust_env_cfg.py','recovery_env_cfg.py','recovery_mdp.py','recovery_math.py',
                   'recovery_state_bank.py','recovery_bank_mdp.py','recovery_back_exploration.py',
                   'recovery_control_targets.py','recovery_smith_math.py','recovery_smith_mdp.py',
                   'recovery_supported_math.py','recovery_supported_mdp.py','__init__.py','agents/rsl_rl_ppo_cfg.py')) {
    $path = Join-Path $installed $name
    $sourceName = switch ($name) { '__init__.py' {'go2_registry_snapshot.py'} 'agents/rsl_rl_ppo_cfg.py' {'go2_ppo_snapshot.py'} default {$name} }
    $sourcePath = Join-Path $portfolio "src/go2_recovery/$sourceName"
    if ((Get-FileHash -LiteralPath $path).Hash -ne (Get-FileHash -LiteralPath $sourcePath).Hash) { throw "Portfolio/installed source mismatch: $name" }
    $sources += @{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}, @{path=$sourcePath;sha256=(Get-FileHash -LiteralPath $sourcePath).Hash}
}
foreach ($path in @((Join-Path $installed 'flat_env_cfg.py'),
                    'E:\IsaacLab\repo\scripts\reinforcement_learning\rsl_rl\train.py',
                    'E:\IsaacLab\env\Lib\site-packages\rsl_rl\runners\on_policy_runner.py',
                    'E:\IsaacLab\env\Lib\site-packages\rsl_rl\algorithms\ppo.py')) {
    $sources += @{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}
}
foreach ($name in @('train.ps1','run_dense_posture_pilot.ps1','check_dense_posture_config.py',
                   'check_smith_standweight_config.py','audit_recovery_report.py','evaluate_recovery.ps1','evaluate_go2_recovery.py')) {
    $path = Join-Path $PSScriptRoot $name
    $sources += @{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}
}
$sources += @{path=$parentCheckpoint;sha256=$parentSha}, @{path=(Join-Path $bank 'states.npz');sha256=$bankSha}
function Assert-Sources {
    foreach ($source in $sources) {
        if ((Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash -ne $source.sha256) { throw "Source/checkpoint/bank drift: $($source.path)" }
    }
}
$outputs = @($snapshotPath,$config,$trainLog)
foreach ($protocol in $protocols) {
    $outputs += (Join-Path $artifacts "$RunTag-$protocol.log"), (Join-Path $portfolio "evaluations/$RunTag-$protocol")
}
foreach ($path in $outputs) { if (Test-Path -LiteralPath $path) { throw "Existing output: $path. Preserve evidence; never restart over partial work." } }
if (@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") }).Count) { throw "Existing training run: $RunTag" }
Write-Output "Dense posture pilot: parent1999; $NumEnvs envs, $Iterations additional updates ($($NumEnvs * $Iterations * 24) environment steps); expected $expectedFinal."
Write-Output 'Standard full-model/optimizer resume; adaptive LR scalar comes from cfg. Not exact uninterrupted simulator/RNG/LR continuation.'
if ($PreflightOnly) { Write-Output 'Preflight passed; no output created or simulator launched.'; return }

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TEMP = 'E:\IsaacLab\tmp'
$env:TMP = 'E:\IsaacLab\tmp'
$env:PYTHONIOENCODING = 'utf-8'
$sources | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $snapshotPath -Encoding UTF8
Push-Location $portfolio
try {
    Assert-Idle
    Assert-Sources
    Write-Output "$(Get-Date -Format o) Starting bounded dense-posture training; log: $trainLog"
    & ./scripts/train.ps1 -Stage recovery_bank_dense_posture -NumEnvs $NumEnvs -MaxIterations $Iterations -LoadRun $parentRun -Checkpoint model_1999.pt -RunName $RunTag -BankPath $bank -Headless *> $trainLog
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $trainLog" }
    Assert-HealthyLog $trainLog
    Assert-Sources
    $runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") })
    if ($runs.Count -ne 1) { throw 'Expected exactly one new training run.' }
    $checkpoint = Join-Path $runs[0].FullName $expectedFinal
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Missing final checkpoint: $checkpoint" }
    New-Item -ItemType Directory -Path $config | Out-Null
    foreach ($name in @('env.yaml','agent.yaml')) { Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$name") -Destination (Join-Path $config $name) }
    & $python ./scripts/check_dense_posture_config.py $config --num-envs $NumEnvs --iterations $Iterations --run-name $RunTag
    if ($LASTEXITCODE -ne 0) { throw 'Saved configuration failed the single-variable guard.' }
    $checkpointSha = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash
    $sources += @{path=$checkpoint;sha256=$checkpointSha}
    Write-Output "$(Get-Date -Format o) Training complete: $checkpoint; SHA256 $checkpointSha"
    foreach ($protocol in $protocols) {
        Assert-Idle
        Assert-Sources
        $directory = Join-Path $portfolio "evaluations/$RunTag-$protocol"
        $evalParams = @{Checkpoint=$checkpoint;Task=$task;OutputDir=$directory;Trials=$Trials;Seed=20260918}
        switch ($protocol) {
            'bank-side' { $evalParams.Poses=@('side'); $evalParams.StateBankPath=$bank; $evalParams.StateBankSplit='heldout'; $evalParams.SettleSeconds=1 }
            'bank-back' { $evalParams.Poses=@('upside_down'); $evalParams.StateBankPath=$bank; $evalParams.StateBankSplit='heldout'; $evalParams.SettleSeconds=1 }
            'upright-pd1' { $evalParams.Poses=@('upright'); $evalParams.SettleSeconds=1; $evalParams.AngleDeg=0 }
            'angle30-side' { $evalParams.Poses=@('side'); $evalParams.AngleDeg=30 }
            'angle30-fore' { $evalParams.Poses=@('fore_aft'); $evalParams.AngleDeg=30 }
        }
        $log = Join-Path $artifacts "$RunTag-$protocol.log"
        Write-Output "$(Get-Date -Format o) Starting isolated scene $protocol; $Trials trials."
        & ./scripts/evaluate_recovery.ps1 @evalParams *> $log
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $protocol" }
        Assert-HealthyLog $log
        Assert-Sources
        $report = Join-Path $directory "$($expectedFinal)_recovery_metrics.json"
        $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
        if ($result.task -ne $task -or $result.seed -ne 20260918 -or
            [IO.Path]::GetFullPath($result.checkpoint) -ne [IO.Path]::GetFullPath($checkpoint) -or
            $result.checkpoint_sha256 -ne $checkpointSha -or $result.self_collisions_enabled -ne $true -or
            $null -ne $result.video_view -or $result.single_policy_acceptance_eligible -eq $false -or
            $result.action_representation.reference -ne 'nominal' -or $result.action_representation.scale -ne 0.25 -or
            $result.action_representation.sample_period_s -ne 0.02 -or $result.action_representation.held_over_physics_substeps -ne 4) {
            throw "Evaluation identity/action metadata mismatch: $report"
        }
        if ((@($result.results.PSObject.Properties.Name) -join ',') -ne ($evalParams.Poses -join ',')) { throw 'Wrong evaluated pose set.' }
        foreach ($poseResult in $result.results.PSObject.Properties.Value) {
            if ($poseResult.trials -ne $Trials -or $poseResult.stable_hold_s -ne 3 -or $poseResult.horizon_s -ne 8) { throw 'Changed trials/hold/horizon.' }
        }
        if ($protocol -like 'bank-*') {
            if ($result.state_bank.sha256 -ne $bankSha -or $result.state_bank.split -ne 'heldout' -or
                $result.settle_requested_s -ne 1 -or $result.settle_actual_s -ne 1 -or
                $result.settle_control_steps -ne 50 -or $null -ne $result.angle_deg) { throw 'Bank protocol mismatch.' }
        } else {
            $settle = if ($protocol -eq 'upright-pd1') { 1 } else { 0 }
            if ($result.angle_deg -ne $evalParams.AngleDeg -or $null -ne $result.state_bank -or
                $result.settle_requested_s -ne $settle -or $result.settle_actual_s -ne $settle -or
                $result.settle_control_steps -ne (50 * $settle)) { throw 'Controlled-drop/nominal-PD protocol mismatch.' }
        }
        & $python ./scripts/audit_recovery_report.py $report
        if ($LASTEXITCODE -ne 0) { throw "Inconsistent report: $report" }
        Write-Output "$(Get-Date -Format o) Completed $protocol. Internal consistency is NOT a standing-performance pass."
    }
    Write-Output 'Finite dense-posture pilot complete; no promotion or automatic extension. Upright-PD tests are compatibility, not fallen recovery.'
}
finally { Pop-Location }
