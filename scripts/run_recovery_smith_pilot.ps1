param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag = '20260916-smithnominal128x2000',
    [ValidateRange(100,3000)][int]$Iterations = 2000,
    [ValidateSet(64,128)][int]$NumEnvs = 128,
    [switch]$PreflightOnly
)
# Finite, single-simulator reward experiment. No model promotion or auto-extension.
$ErrorActionPreference = 'Stop'
$portfolio = Split-Path $PSScriptRoot -Parent
$runRoot = 'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery'
$artifactRoot = 'E:\IsaacLab\artifacts\recovery-20260916'
$installedRoot = 'E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$bankPath = Join-Path $portfolio 'datasets/recovery_states/nominal_pd_v1_20260916'
$bootstrapRun = 'bootstrap_fresh42_target_v1'
$bootstrap = Join-Path $runRoot "$bootstrapRun/model_0.pt"
$trainLog = Join-Path $artifactRoot "$RunTag-train.log"
$configDir = Join-Path $portfolio "configs/$RunTag"
if ((Get-FileHash -LiteralPath $bootstrap -Algorithm SHA256).Hash -ne 'DAA28DD10EF200121BDD6F2A14A03FCD856859EBB5F3EF510D5D946A10F11CB9') {
    throw 'Fresh seed42 bootstrap hash mismatch.'
}
if ((Get-FileHash -LiteralPath (Join-Path $bankPath 'states.npz') -Algorithm SHA256).Hash -ne '71B882E92035B4C248E5980339C980DBB242DCB1EC711C05252C33249DB48F5A') {
    throw 'State bank hash mismatch.'
}
$activeSim = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'E:[/\\]IsaacLab' -and
    $_.CommandLine -match 'rsl_rl[/\\]train.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py'
})
if ($activeSim.Count -gt 0) { throw "Another simulator is active: $($activeSim.ProcessId -join ',')" }
$paths = @($trainLog, $configDir, (Join-Path $artifactRoot "$RunTag-source-snapshot.json"))
foreach ($protocol in @('heldout','angle30','angle45')) {
    $paths += Join-Path $portfolio "evaluations/$RunTag-$protocol"
    $paths += Join-Path $artifactRoot "$RunTag-$protocol.log"
}
foreach ($path in $paths) {
    if (Test-Path -LiteralPath $path) { throw "Existing output $path; do not overwrite or restart." }
}
if (@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") }).Count -gt 0) {
    throw 'A training run with this tag already exists.'
}
$sourceSnapshot = @()
foreach ($name in @('recovery_smith_math.py','recovery_smith_mdp.py','recovery_env_cfg.py',
                   'recovery_bank_mdp.py','recovery_state_bank.py','recovery_math.py','recovery_mdp.py',
                   'recovery_control_targets.py','flat_env_cfg.py','__init__.py','agents/rsl_rl_ppo_cfg.py')) {
    $path = Join-Path $installedRoot $name
    $sourceSnapshot += @{ path=$path; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
}
foreach ($name in @('recovery_smith_math.py','recovery_smith_mdp.py','recovery_env_cfg.py')) {
    if ((Get-FileHash -LiteralPath (Join-Path $installedRoot $name)).Hash -ne
        (Get-FileHash -LiteralPath (Join-Path $portfolio "src/go2_recovery/$name")).Hash) {
        throw "Portfolio/installed code mismatch: $name"
    }
}
function Assert-PhysicsLogHealthy([string]$LogPath) {
    $bad = Select-String -LiteralPath $LogPath -Pattern 'CUDA error|PhysX error|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($null -ne $bad) { throw "Invalid simulation log $LogPath : $($bad.Line)" }
}
Write-Output "SmithNominal reward-only pilot: $Iterations iterations, $NumEnvs environments, $($Iterations * $NumEnvs * 24) environment steps."
if ($PreflightOnly) { Write-Output 'Preflight passed; no output created and no simulator launched.'; return }
Push-Location $portfolio
try {
    $env:PYTHONUNBUFFERED = '1'
    $sourceSnapshot | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $artifactRoot "$RunTag-source-snapshot.json") -Encoding UTF8
    Write-Output "$(Get-Date -Format o) Starting fresh reward-only training from seed42; see $trainLog"
    & ./scripts/train.ps1 -Stage recovery_bank_smith_nominal -NumEnvs $NumEnvs -MaxIterations $Iterations -LoadRun $bootstrapRun -Checkpoint model_0.pt -RunName $RunTag -BankPath $bankPath -Headless *> $trainLog
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $trainLog" }
    Assert-PhysicsLogHealthy $trainLog
    foreach ($source in $sourceSnapshot) {
        if ((Get-FileHash -LiteralPath $source.path).Hash -ne $source.sha256) { throw 'Installed sources changed during training.' }
    }
    $runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") })
    if ($runs.Count -ne 1) { throw 'Expected exactly one pilot run directory.' }
    $checkpoint = Join-Path $runs[0].FullName "model_$($Iterations - 1).pt"
    if (-not (Test-Path -LiteralPath $checkpoint)) { throw 'Expected final checkpoint is missing.' }
    New-Item -ItemType Directory -Path $configDir | Out-Null
    foreach ($name in @('env.yaml','agent.yaml')) {
        Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$name") -Destination (Join-Path $configDir $name)
    }
    & E:\IsaacLab\env\python.exe ./scripts/check_smith_experiment_config.py $configDir --num-envs $NumEnvs --iterations $Iterations
    if ($LASTEXITCODE -ne 0) { throw 'Formal saved configuration is not the intended reward-only comparison.' }
    Write-Output "$(Get-Date -Format o) Training complete: $checkpoint; SHA256 $((Get-FileHash -LiteralPath $checkpoint).Hash)"
    foreach ($protocol in @('heldout','angle30','angle45')) {
        $evalArgs = @{
            Checkpoint=$checkpoint; Task='Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0'
            OutputDir=(Join-Path $portfolio "evaluations/$RunTag-$protocol"); Trials=20
        }
        if ($protocol -eq 'heldout') {
            $evalArgs.StateBankPath=$bankPath; $evalArgs.SettleSeconds=1
            $evalArgs.Seed=20260918; $evalArgs.Poses=@('side','upside_down')
        } else {
            $evalArgs.AngleDeg=if ($protocol -eq 'angle30') { 30 } else { 45 }
            $evalArgs.Seed=if ($protocol -eq 'angle30') { 20260918 } else { 20260916 }
            $evalArgs.Poses=if ($protocol -eq 'angle30') { @('upright','side','fore_aft') } else { @('upright','side','fore_aft','upside_down') }
        }
        $evalLog=Join-Path $artifactRoot "$RunTag-$protocol.log"
        & ./scripts/evaluate_recovery.ps1 @evalArgs *> $evalLog
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $protocol" }
        Assert-PhysicsLogHealthy $evalLog
        Write-Output "$(Get-Date -Format o) Completed $protocol; inspect strict actual outcomes before promotion."
    }
    Write-Output 'Finite SmithNominal pilot complete. No automatic promotion or further training.'
}
finally { Pop-Location }
