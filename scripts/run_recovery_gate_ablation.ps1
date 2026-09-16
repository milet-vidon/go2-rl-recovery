param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag,
    [string]$LoadRun = '2026-09-16_13-16-04_bank_nominalpd_from3900_20260916',
    [string]$Checkpoint = 'model_4899.pt',
    [ValidateRange(1,400)][int]$Iterations = 100,
    [string]$BankPath = 'E:/IsaacLab/go2-rl-open-source/datasets/recovery_states/nominal_pd_v1_20260916'
)
# A finite, serial two-arm experiment; never loops until a reward threshold.
# Both arms load exactly the same actor/critic, Adam state and exploration std.
$ErrorActionPreference = 'Stop'
$portfolio = Split-Path $PSScriptRoot -Parent
$runRoot = 'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery'
$artifactRoot = 'E:\IsaacLab\artifacts\recovery-20260916'
$parentCheckpoint = Join-Path (Join-Path $runRoot $LoadRun) $Checkpoint
if (-not (Test-Path -LiteralPath $parentCheckpoint -PathType Leaf)) { throw "Missing parent: $parentCheckpoint" }
if (-not (Test-Path -LiteralPath $BankPath)) { throw "Missing bank: $BankPath" }
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($portfolio)) -ne 'E:\') { throw 'Portfolio must be on E:.' }
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($BankPath)) -ne 'E:\') { throw 'Bank must be on E:.' }
$activeSim = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'E:[/\\]IsaacLab' -and
    $_.CommandLine -match 'rsl_rl[/\\]train.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py'
})
if ($activeSim.Count -gt 0) { throw "Another local simulator is running (PID $($activeSim.ProcessId -join ',')). Do not duplicate." }
foreach ($arm in @('control','release')) {
    foreach ($relative in @("configs/$RunTag-$arm", "evaluations/$RunTag-$arm-heldout", "evaluations/$RunTag-$arm-angle30", "evaluations/$RunTag-$arm-angle45")) {
        if (Test-Path -LiteralPath (Join-Path $portfolio $relative)) { throw "Existing output: $relative. Inspect it, do not overwrite/restart." }
    }
    if (Test-Path -LiteralPath (Join-Path $artifactRoot "$RunTag-$arm-train.log")) { throw "Existing run log for $arm. Inspect before resuming." }
}
Write-Output "Parent SHA256: $((Get-FileHash -LiteralPath $parentCheckpoint -Algorithm SHA256).Hash)"
Write-Output "Bank SHA256: $((Get-FileHash -LiteralPath (Join-Path $BankPath 'states.npz') -Algorithm SHA256).Hash)"
Push-Location $portfolio
try {
    foreach ($arm in @('control','release')) {
        $runName = "$RunTag`_$arm"
        $trainLog = Join-Path $artifactRoot "$RunTag-$arm-train.log"
        Write-Output "Starting $arm from $parentCheckpoint for $Iterations additional iterations."
        & ./scripts/train.ps1 -Stage "recovery_bank_$arm" -NumEnvs 512 -MaxIterations $Iterations -LoadRun $LoadRun -Checkpoint $Checkpoint -RunName $runName -BankPath $BankPath -Headless *> $trainLog
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $trainLog" }
        $runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$runName") })
        if ($runs.Count -ne 1) { throw "Expected one new $runName directory, found $($runs.Count)" }
        $candidate = Get-ChildItem -LiteralPath $runs[0].FullName -Filter 'model_*.pt' |
            Sort-Object { [int]($_.BaseName -replace '^model_', '') } | Select-Object -Last 1
        if ($null -eq $candidate) { throw "No checkpoint in $($runs[0].FullName)" }
        $configDir = Join-Path $portfolio "configs/$RunTag-$arm"
        New-Item -ItemType Directory -Path $configDir | Out-Null
        foreach ($configName in @('env.yaml','agent.yaml')) {
            Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$configName") -Destination (Join-Path $configDir $configName)
        }
        Write-Output "$arm final: $($candidate.FullName); SHA256 $((Get-FileHash -LiteralPath $candidate.FullName -Algorithm SHA256).Hash)"
        foreach ($protocol in @('heldout','angle30','angle45')) {
            $evalLog = Join-Path $artifactRoot "$RunTag-$arm-$protocol.log"
            $evalArgs = @{
                Checkpoint = $candidate.FullName
                OutputDir = Join-Path $portfolio "evaluations/$RunTag-$arm-$protocol"
                Trials = 20
            }
            if ($protocol -eq 'heldout') {
                $evalArgs.Task = 'Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0'
                $evalArgs.StateBankPath = $BankPath
                $evalArgs.StateBankSplit = 'heldout'
                $evalArgs.SettleSeconds = 1
                $evalArgs.Seed = 20260918
                $evalArgs.Poses = @('side','upside_down')
            } else {
                $evalArgs.Task = 'Isaac-Recovery-Aligned-Flat-Unitree-Go2-Play-v0'
                $evalArgs.AngleDeg = if ($protocol -eq 'angle30') { 30 } else { 45 }
                $evalArgs.Seed = if ($protocol -eq 'angle30') { 20260918 } else { 20260916 }
                $evalArgs.Poses = if ($protocol -eq 'angle30') { @('upright','side','fore_aft') } else { @('upright','side','fore_aft','upside_down') }
            }
            & ./scripts/evaluate_recovery.ps1 @evalArgs *> $evalLog
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $evalLog" }
            Write-Output "Completed $arm $protocol; inspect metrics before promotion or further training."
        }
    }
    Write-Output 'Paired experiment complete. No model is automatically promoted and no next training is automatically launched.'
}
finally { Pop-Location }
