param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag = '20260916-smithstandpair128x1000',
    [ValidateSet(2,1000)][int]$Iterations = 1000,
    [ValidateSet(16,128)][int]$NumEnvs = 128,
    [switch]$PreflightOnly
)
# Finite paired continuation: only gated stand reward weight differs (10 vs 30).
# Both arms load the same complete model/optimizer state. Never auto-promote.
$ErrorActionPreference = 'Stop'
if (($Iterations -eq 2) -ne ($NumEnvs -eq 16)) { throw 'Use 16 envs/2 iterations for smoke or 128/1000 for the formal pair.' }
$portfolio = Split-Path $PSScriptRoot -Parent
$artifacts = 'E:\IsaacLab\artifacts\recovery-20260916'
$runRoot = 'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery'
$parentRun = '2026-09-16_19-02-40_20260916-smithnominal128x2000'
$parentCheckpoint = Join-Path $runRoot "$parentRun/model_1999.pt"
$parentSha = '71E4C7A39FF81836D2EAD1C4533988C696E5BD4A0FC939B8CEF6B4F6DAEC1B1C'
$bank = Join-Path $portfolio 'datasets/recovery_states/nominal_pd_v1_20260916'
$bankSha = '71B882E92035B4C248E5980339C980DBB242DCB1EC711C05252C33249DB48F5A'
$task = 'Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0'
$python = 'E:\IsaacLab\env\python.exe'
$snapshotPath = Join-Path $artifacts "$RunTag-source-snapshot.json"
$expectedFinal = "model_$(1999 + $Iterations - 1).pt"

function Assert-Idle {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pythonw.exe','kit.exe','isaac-sim.exe') -and
        $_.CommandLine -match 'E:[/\\]IsaacLab' -and
        ($_.Name -in @('kit.exe','isaac-sim.exe') -or $_.CommandLine -match 'rsl_rl[/\\]train.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py')
    })
    if ($active.Count) { throw "Another simulator is active: $($active.ProcessId -join ',')." }
}
function Assert-HealthyLog([string]$Path) {
    $bad = Select-String -LiteralPath $Path -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($bad) { throw "Invalid simulation: $Path : $($bad.Line)" }
}
Assert-Idle
if ((Get-FileHash -LiteralPath $parentCheckpoint).Hash -ne $parentSha) { throw 'Parent checkpoint mismatch.' }
if ((Get-FileHash -LiteralPath (Join-Path $bank 'states.npz')).Hash -ne $bankSha) { throw 'Bank mismatch.' }
foreach ($name in @('env.yaml','agent.yaml')) {
    $actual = Join-Path $runRoot "$parentRun/params/$name"
    $archive = Join-Path $portfolio "configs/20260916-smithnominal128x2000/$name"
    if ((Get-FileHash -LiteralPath $actual).Hash -ne (Get-FileHash -LiteralPath $archive).Hash) { throw "Parent config archive mismatch: $name" }
}
$sources = @(Get-Content -LiteralPath (Join-Path $artifacts '20260916-smithnominal128x2000-source-snapshot.json') -Raw | ConvertFrom-Json)
if ($sources.Count -ne 11) { throw 'Expected original 11 installed-source records.' }
foreach ($source in $sources) {
    if ((Get-FileHash -LiteralPath $source.path).Hash -ne $source.sha256) { throw "Installed source drift: $($source.path)" }
}
foreach ($name in @('train.ps1','run_smith_standweight_pair.ps1','check_smith_standweight_config.py','audit_recovery_report.py','evaluate_recovery.ps1','evaluate_go2_recovery.py')) {
    $path = Join-Path $PSScriptRoot $name
    $sources += @{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}
}
$sources += @{path=$parentCheckpoint;sha256=$parentSha}, @{path=(Join-Path $bank 'states.npz');sha256=$bankSha}
function Assert-Sources {
    foreach ($source in $sources) {
        if ((Get-FileHash -LiteralPath $source.path).Hash -ne $source.sha256) { throw "Source/checkpoint/bank drift: $($source.path)" }
    }
}
$outputs = @($snapshotPath)
foreach ($weight in @(10,30)) {
    $tag = "${RunTag}_w$weight"
    if (@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$tag") }).Count) { throw "Existing run: $tag" }
    $outputs += (Join-Path $artifacts "$tag-train.log"), (Join-Path $portfolio "configs/$tag")
    if ($Iterations -ne 2) {
        foreach ($protocol in @('heldout','angle30','angle45')) {
            $outputs += (Join-Path $artifacts "$tag-$protocol.log"), (Join-Path $portfolio "evaluations/$tag-$protocol")
        }
    }
}
foreach ($path in $outputs) { if (Test-Path -LiteralPath $path) { throw "Existing output: $path. Preserve partial work and resume only missing stages manually." } }
Write-Output "Smith stand-weight pair: w10 vs w30, SAME parent1999, $NumEnvs envs, $Iterations extra iterations per arm; expected $expectedFinal."
if ($PreflightOnly) { Write-Output 'Preflight passed; no files created or simulator started.'; return }

$env:PYTHONDONTWRITEBYTECODE = '1'; $env:PYTHONUNBUFFERED = '1'
$env:TEMP = 'E:\IsaacLab\tmp'; $env:TMP = 'E:\IsaacLab\tmp'; $env:PYTHONIOENCODING = 'utf-8'
$sources | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $snapshotPath -Encoding UTF8
Push-Location $portfolio
try {
    foreach ($weight in @(10,30)) {
        Assert-Idle; Assert-Sources
        $tag = "${RunTag}_w$weight"
        $trainLog = Join-Path $artifacts "$tag-train.log"
        Write-Output "$(Get-Date -Format o) Starting $tag from the same complete parent checkpoint."
        & ./scripts/train.ps1 -Stage recovery_bank_smith_nominal -NumEnvs $NumEnvs -MaxIterations $Iterations -LoadRun $parentRun -Checkpoint model_1999.pt -RunName $tag -BankPath $bank -SmithStandWeight $weight -Headless *> $trainLog
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $trainLog" }
        Assert-HealthyLog $trainLog; Assert-Sources
        $runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$tag") })
        if ($runs.Count -ne 1) { throw "Expected exactly one run for $tag" }
        $checkpoint = Join-Path $runs[0].FullName $expectedFinal
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Missing final checkpoint: $checkpoint" }
        $config = Join-Path $portfolio "configs/$tag"
        New-Item -ItemType Directory -Path $config | Out-Null
        foreach ($name in @('env.yaml','agent.yaml')) { Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$name") -Destination (Join-Path $config $name) }
        & $python ./scripts/check_smith_standweight_config.py $config --num-envs $NumEnvs --iterations $Iterations --stand-weight $weight
        if ($LASTEXITCODE -ne 0) { throw "Saved configuration failed the single-variable guard: $tag" }
        Write-Output "$(Get-Date -Format o) Completed $tag training; SHA256 $((Get-FileHash -LiteralPath $checkpoint).Hash)"
        if ($Iterations -eq 2) { Write-Output 'Two-iteration smoke: validates continuation/override/config only, not recovery performance.'; continue }
        foreach ($protocol in @('heldout','angle30','angle45')) {
            Assert-Idle; Assert-Sources
            $directory = Join-Path $portfolio "evaluations/$tag-$protocol"
            $evalParams = @{Checkpoint=$checkpoint;Task=$task;OutputDir=$directory;Trials=20}
            if ($protocol -eq 'heldout') {
                $evalParams.StateBankPath=$bank; $evalParams.StateBankSplit='heldout'; $evalParams.SettleSeconds=1
                $evalParams.Seed=20260918; $evalParams.Poses=@('side','upside_down')
            } else {
                $evalParams.AngleDeg=if($protocol -eq 'angle30'){30}else{45}
                $evalParams.Seed=if($protocol -eq 'angle30'){20260918}else{20260916}
                $evalParams.Poses=if($protocol -eq 'angle30'){@('upright','side','fore_aft')}else{@('upright','side','fore_aft','upside_down')}
            }
            $log = Join-Path $artifacts "$tag-$protocol.log"
            & ./scripts/evaluate_recovery.ps1 @evalParams *> $log
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $tag/$protocol" }
            Assert-HealthyLog $log
            $report = Join-Path $directory "$($expectedFinal)_recovery_metrics.json"
            $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
            if ($result.task -ne $task -or $result.seed -ne $evalParams.Seed -or
                [IO.Path]::GetFullPath($result.checkpoint) -ne [IO.Path]::GetFullPath($checkpoint) -or
                $result.checkpoint_sha256 -ne (Get-FileHash -LiteralPath $checkpoint).Hash -or
                $result.self_collisions_enabled -ne $true -or $null -ne $result.video_view -or
                $result.action_representation.reference -ne 'nominal' -or $result.action_representation.scale -ne 0.25 -or
                $result.action_representation.sample_period_s -ne 0.02 -or $result.action_representation.held_over_physics_substeps -ne 4) {
                throw "Evaluation identity or action metadata mismatch: $report"
            }
            if ((@($result.results.PSObject.Properties.Name | Sort-Object) -join ',') -ne (@($evalParams.Poses | Sort-Object) -join ',')) { throw 'Wrong evaluated pose set.' }
            foreach ($poseResult in $result.results.PSObject.Properties.Value) {
                if ($poseResult.trials -ne 20 -or $poseResult.stable_hold_s -ne 3 -or $poseResult.horizon_s -ne 8) { throw 'Changed trial count/hold/horizon.' }
            }
            if ($protocol -eq 'heldout') {
                if ($result.state_bank.sha256 -ne $bankSha -or $result.state_bank.split -ne 'heldout' -or
                    $result.settle_requested_s -ne 1 -or $result.settle_actual_s -ne 1 -or
                    $result.settle_control_steps -ne 50 -or $null -ne $result.angle_deg) { throw 'Bank protocol mismatch.' }
            } elseif ($result.angle_deg -ne $evalParams.AngleDeg -or $result.settle_actual_s -ne 0 -or $null -ne $result.state_bank) { throw 'Controlled-drop protocol mismatch.' }
            & $python ./scripts/audit_recovery_report.py $report
            if ($LASTEXITCODE -ne 0) { throw "Inconsistent report: $report" }
            Write-Output "$(Get-Date -Format o) Completed $tag/$protocol. Consistency is NOT a performance pass."
        }
    }
    Write-Output 'Finite stand-weight pair completed. Compare final standing, not total reward; no promotion or auto-extension.'
}
finally { Pop-Location }
