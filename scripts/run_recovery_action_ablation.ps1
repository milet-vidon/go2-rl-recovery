param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag,
    [ValidateRange(100,3000)][int]$Iterations = 1000
)
# Finite serial experiment. Fresh matched weights, NOT continuation of an old actor.
$ErrorActionPreference = 'Stop'
$portfolio = Split-Path $PSScriptRoot -Parent
$runRoot = 'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery'
$artifactRoot = 'E:\IsaacLab\artifacts\recovery-20260916'
$bankPath = Join-Path $portfolio 'datasets/recovery_states/nominal_pd_v1_20260916'
$bootstrapRun = 'bootstrap_fresh42_target_v1'
$bootstrap = Join-Path $runRoot "$bootstrapRun/model_0.pt"
if ((Get-FileHash -LiteralPath $bootstrap -Algorithm SHA256).Hash -ne 'DAA28DD10EF200121BDD6F2A14A03FCD856859EBB5F3EF510D5D946A10F11CB9') {
    throw 'Fresh seed bootstrap hash mismatch; never silently resume another checkpoint.'
}
if ((Get-FileHash -LiteralPath (Join-Path $bankPath 'states.npz') -Algorithm SHA256).Hash -ne '71B882E92035B4C248E5980339C980DBB242DCB1EC711C05252C33249DB48F5A') {
    throw 'Nominal-PD bank hash mismatch.'
}
$activeSim = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'E:[/\\]IsaacLab' -and
    $_.CommandLine -match 'rsl_rl[/\\]train.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py'
})
if ($activeSim.Count -gt 0) { throw "Another simulator is active: $($activeSim.ProcessId -join ',')" }
foreach ($arm in @('nominal','current')) {
    foreach ($path in @("configs/$RunTag-$arm", "evaluations/$RunTag-$arm-heldout", "evaluations/$RunTag-$arm-angle30", "evaluations/$RunTag-$arm-angle45")) {
        if (Test-Path -LiteralPath (Join-Path $portfolio $path)) { throw "Existing output $path; do not restart." }
    }
    if (Test-Path -LiteralPath (Join-Path $artifactRoot "$RunTag-$arm-train.log")) { throw "Existing $arm log; do not overwrite." }
}
Push-Location $portfolio
try {
    foreach ($arm in @('nominal','current')) {
        $runName = "$RunTag`_$arm"
        $trainLog = Join-Path $artifactRoot "$RunTag-$arm-train.log"
        Write-Output "$(Get-Date -Format o) Starting ${arm}: $Iterations fresh iterations, 512 environments."
        & ./scripts/train.ps1 -Stage "recovery_bank_$($arm)_target" -NumEnvs 512 -MaxIterations $Iterations -LoadRun $bootstrapRun -Checkpoint model_0.pt -RunName $runName -BankPath $bankPath -Headless *> $trainLog
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $trainLog" }
        $runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$runName") })
        if ($runs.Count -ne 1) { throw 'Expected exactly one formal run directory.' }
        $candidate = Get-ChildItem -LiteralPath $runs[0].FullName -Filter 'model_*.pt' |
            Sort-Object { [int]($_.BaseName -replace '^model_', '') } | Select-Object -Last 1
        if ($null -eq $candidate -or $candidate.BaseName -ne "model_$($Iterations - 1)") { throw 'Missing expected final checkpoint.' }
        $configDir = Join-Path $portfolio "configs/$RunTag-$arm"
        New-Item -ItemType Directory -Path $configDir | Out-Null
        foreach ($name in @('env.yaml','agent.yaml')) {
            Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$name") -Destination (Join-Path $configDir $name)
        }
        Write-Output "$(Get-Date -Format o) $arm checkpoint: $($candidate.FullName); SHA256 $((Get-FileHash -LiteralPath $candidate.FullName).Hash)"
        $variant = if ($arm -eq 'nominal') { 'NominalTarget' } else { 'CurrentTarget' }
        foreach ($protocol in @('heldout','angle30','angle45')) {
            $evalArgs = @{
                Checkpoint = $candidate.FullName
                Task = "Isaac-Recovery-Bank-$variant-Flat-Unitree-Go2-Play-v0"
                OutputDir = Join-Path $portfolio "evaluations/$RunTag-$arm-$protocol"
                Trials = 20
            }
            if ($protocol -eq 'heldout') {
                $evalArgs.StateBankPath = $bankPath
                $evalArgs.SettleSeconds = 1
                $evalArgs.Seed = 20260918
                $evalArgs.Poses = @('side','upside_down')
            } else {
                $evalArgs.AngleDeg = if ($protocol -eq 'angle30') { 30 } else { 45 }
                $evalArgs.Seed = if ($protocol -eq 'angle30') { 20260918 } else { 20260916 }
                $evalArgs.Poses = if ($protocol -eq 'angle30') { @('upright','side','fore_aft') } else { @('upright','side','fore_aft','upside_down') }
            }
            & ./scripts/evaluate_recovery.ps1 @evalArgs *> (Join-Path $artifactRoot "$RunTag-$arm-$protocol.log")
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $arm $protocol" }
            Write-Output "$(Get-Date -Format o) Completed $arm $protocol. Inspect actual pose results; no automatic promotion."
        }
    }
    Write-Output 'Action-reference experiment finished; inspect results before any continuation. Good historical models are untouched.'
}
finally { Pop-Location }
