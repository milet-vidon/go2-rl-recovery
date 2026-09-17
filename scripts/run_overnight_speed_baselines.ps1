param(
    [ValidatePattern('^20260917-speed-baselines(?:-[A-Za-z0-9_-]+)?$')]
    [string]$RunTag = '20260917-speed-baselines',
    [switch]$PreflightOnly
)
# Finite read-only checkpoint probe, NOT training, candidate promotion or running acceptance.
# Run only after the current recovery orchestrator and all its evaluations have exited.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$portfolio = Split-Path $PSScriptRoot -Parent
$python = 'E:\IsaacLab\env\python.exe'
$task = 'Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0'
$evaluator = Join-Path $PSScriptRoot 'evaluate_stand_walk_stop.ps1'
$verifier = Join-Path $PSScriptRoot 'verify_overnight_baselines.py'
$outputRoot = Join-Path $portfolio "evaluations/$RunTag"
$logRoot = Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
$summaryPath = Join-Path $outputRoot 'summary.json'
$seeds = @(20260909,20260910,20260911)
$speeds = @('0.5','0.8','1.0','1.5','2.0')
$models = @(
    @{id='locomotion2250'; relative='models/locomotion/natural_stop_diagonal_model2250.pt'; sha256='33be609b3daab4a2f773d70aeeb41bc42c5cbc843780efc963ed04d5dd368417'},
    @{id='robust3648'; relative='models/locomotion/natural_robust_push_model3648.pt'; sha256='475f07bf0b05101dd4157212b4e3fd48a04731d452d7a349fe467c508dbaaac8'}
)

function Assert-EPath([string]$Path) {
    if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)) -ne 'E:\') {
        throw "Non-E-drive path: $Path"
    }
}

function Assert-Idle {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pythonw.exe','kit.exe','isaac-sim.exe') -and
        $_.CommandLine -match 'E:[/\\]IsaacLab' -and
        ($_.Name -in @('kit.exe','isaac-sim.exe') -or $_.CommandLine -match 'rsl_rl[/\\]train.py|resume_smith_checkpoint_entry.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py')
    })
    if ($active.Count) { throw "Another simulator is active: $($active.ProcessId -join ','). Do not overlap recovery training/evaluation." }
}

function Assert-PreservedBaselines {
    & $python $verifier | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw 'Exported baseline preservation check failed.' }
    foreach ($model in $models) {
        if ((Get-FileHash -LiteralPath $model.path -Algorithm SHA256).Hash -ne $model.sha256) {
            throw "Frozen speed-probe checkpoint changed: $($model.path)"
        }
    }
}

function Assert-HealthyLog([string]$Path) {
    $bad = Select-String -LiteralPath $Path -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|\[Error\].*PhysX|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($null -ne $bad) { throw "Invalid simulation: $Path : $($bad.Line)" }
}

function Assert-FiniteNumber($Value, [string]$Field) {
    if ($null -eq $Value -or $Value -is [string] -or $Value -is [bool]) { throw "Invalid numeric report field: $Field" }
    $number = [double]$Value
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "Nonfinite report field: $Field" }
}

foreach ($path in @($portfolio,$python,$evaluator,$verifier,$outputRoot,$logRoot)) { Assert-EPath $path }
foreach ($path in @($python,$evaluator,$verifier,(Join-Path $portfolio 'configs/overnight_preservation_20260917.json'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing dependency: $path" }
}
foreach ($model in $models) {
    $model.path = Join-Path $portfolio $model.relative
    Assert-EPath $model.path
    if (-not (Test-Path -LiteralPath $model.path -PathType Leaf)) { throw "Missing baseline: $($model.path)" }
}
foreach ($path in @($outputRoot,$logRoot)) {
    if (Test-Path -LiteralPath $path) { throw "Existing output: $path. Preserve partial evidence; use a fresh suffixed RunTag." }
}
Assert-Idle
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = 'E:\IsaacLab\tmp'
$env:TMP = 'E:\IsaacLab\tmp'
Assert-PreservedBaselines
$sourcePaths = @($evaluator,(Join-Path $PSScriptRoot 'evaluate_go2_stand_walk_stop.py'),$verifier,$PSCommandPath)
$sources = @($sourcePaths | ForEach-Object { @{path=$_; sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash} })
function Assert-Sources {
    foreach ($source in $sources) {
        if ((Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash -ne $source.sha256) {
            throw "Evaluation source changed during this finite probe: $($source.path)"
        }
    }
}
Write-Host 'Frozen 2250/3648 diagnostic only: at most 30 serial rollouts; no training or promotion.'
Write-Host 'Speed ladder: 0.5, 0.8, 1.0, 1.5, 2.0 m/s. Three existing-report passes are required to advance each model.'
Write-Host 'Separate drift gates (abs mean vy < 0.12, abs mean wz < 0.15) prevent a promotion claim, but do not stop diagnostic speed probing.'
if ($PreflightOnly) { Write-Host 'Preflight passed. No output created or simulator launched.'; return }

New-Item -ItemType Directory -Path $outputRoot | Out-Null
New-Item -ItemType Directory -Path $logRoot | Out-Null
$summary = [ordered]@{
    schema_version='overnight_speed_baselines_v1'; run_tag=$RunTag; status='running'; started_at=(Get-Date -Format o)
    diagnostic_only=$true; training_performed=$false; promotion_performed=$false; acceptance_eligible=$false
    task=$task; seeds=$seeds; commanded_speeds_m_s=$speeds; models=$models; source_snapshot=$sources
    speed_advance_rule='All three seeds must pass the unchanged report.passed at the previous speed. Failure skips higher speeds for that model.'
    separate_drift_rule='abs(settled_phase_stats.walk.vy_b_mean) < 0.12 m/s and abs(settled_phase_stats.walk.yaw_rate_mean) < 0.15 rad/s. Failed drift blocks any promotion claim, not diagnostic speed escalation.'
    limitations='Simulation baseline probes only; no recovered walking or integrated controller test; commanded 2.0 m/s is not evidence of actual fast running. Every actual speed and failure remains in the reports.'
    completed_rollouts=@(); completed_levels=@(); skipped_levels=@()
}
function Save-Summary {
    $summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
}
$failure = $null
$preservationFailure = $null
Save-Summary
try {
    foreach ($model in $models) {
        $advance = $true
        foreach ($speedText in $speeds) {
            $speed = [double]::Parse($speedText,[Globalization.CultureInfo]::InvariantCulture)
            if (-not $advance) {
                $summary.skipped_levels += @{model=$model.id; commanded_speed_m_s=$speed; reason='Previous speed failed unchanged report.passed in at least one seed.'}
                Save-Summary
                continue
            }
            $levelRows = @()
            foreach ($seed in $seeds) {
                Assert-Idle
                Assert-Sources
                Assert-PreservedBaselines
                $caseName = "$($model.id)-speed$($speedText.Replace('.','p'))-seed$seed"
                $caseDir = Join-Path $outputRoot $caseName
                $log = Join-Path $logRoot "$caseName.log"
                if ((Test-Path -LiteralPath $caseDir) -or (Test-Path -LiteralPath $log)) { throw "Existing case output: $caseName" }
                Write-Host "$(Get-Date -Format o) Starting $caseName"
                & $evaluator -Checkpoint $model.path -Task $task -OutputDir $caseDir -Seed $seed -WalkSpeed $speed -LateralSpeed 0 -YawRate 0 -PushSpeed 0 -NoVideo *> $log
                if ($LASTEXITCODE -ne 0) { throw "Evaluation command failed: $log" }
                Assert-HealthyLog $log
                Assert-Sources
                Assert-PreservedBaselines
                $reportPath = Join-Path $caseDir (([IO.Path]::GetFileNameWithoutExtension($model.path)) + '_stand_walk_stop.json')
                if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw "Missing actual report: $reportPath" }
                $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
                if ($report.protocol_version -ne 'stand_walk_stop_stance_geometry_v2' -or $report.task -ne $task -or
                    $report.seed -ne $seed -or $report.video_view -ne $null -or
                    [IO.Path]::GetFullPath($report.checkpoint) -ne [IO.Path]::GetFullPath($model.path) -or
                    $report.checkpoint_sha256 -ne $model.sha256 -or $report.passed -isnot [bool]) {
                    throw "Report identity/protocol mismatch: $reportPath"
                }
                if ($report.protocol.stand_s -ne 4 -or $report.protocol.walk_s -ne 8 -or $report.protocol.stop_s -ne 6 -or
                    $report.protocol.walk_speed -ne $speed -or $report.protocol.lateral_speed -ne 0 -or
                    $report.protocol.yaw_rate -ne 0 -or $report.protocol.push_delta_vy -ne 0) {
                    throw "Unexpected command protocol: $reportPath"
                }
                $acceptance = @($report.acceptance.PSObject.Properties)
                if ($acceptance.Count -eq 0 -or @($acceptance | Where-Object { $_.Value -isnot [bool] }).Count) { throw "Invalid acceptance fields: $reportPath" }
                $failedCriteria = @($acceptance | Where-Object { $_.Value -eq $false } | ForEach-Object { $_.Name })
                if ($report.passed -ne ($failedCriteria.Count -eq 0)) { throw "Inconsistent passed flag: $reportPath" }
                $walk = $report.settled_phase_stats.walk
                foreach ($field in @('vx_b_mean','vy_b_mean','yaw_rate_mean','max_tilt_deg','contact_slip_mean')) {
                    Assert-FiniteNumber $walk.$field "walk.$field"
                }
                $lateralPass = [Math]::Abs([double]$walk.vy_b_mean) -lt 0.12
                $yawPass = [Math]::Abs([double]$walk.yaw_rate_mean) -lt 0.15
                $row = [ordered]@{
                    model=$model.id; checkpoint=$model.path; checkpoint_sha256=$model.sha256
                    seed=$seed; commanded_speed_m_s=$speed; measured_vx_b_mean_m_s=$walk.vx_b_mean
                    measured_vy_b_mean_m_s=$walk.vy_b_mean; measured_yaw_rate_mean_rad_s=$walk.yaw_rate_mean
                    max_walk_tilt_deg=$walk.max_tilt_deg; mean_contact_slip_m_s=$walk.contact_slip_mean
                    report_passed=$report.passed; report_failed_criteria=$failedCriteria
                    separate_lateral_drift_passed=$lateralPass; separate_yaw_drift_passed=$yawPass
                    report_and_separate_drift_passed=($report.passed -and $lateralPass -and $yawPass)
                    report=$reportPath; report_sha256=(Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash; log=$log
                }
                $levelRows += $row
                $summary.completed_rollouts += $row
                Save-Summary
                Write-Host "$caseName : report_passed=$($report.passed), measured vx=$($walk.vx_b_mean), drift_passed=$($lateralPass -and $yawPass)"
            }
            $advance = @($levelRows | Where-Object { -not $_.report_passed }).Count -eq 0
            $summary.completed_levels += @{
                model=$model.id; commanded_speed_m_s=$speed; seeds_completed=$levelRows.Count
                all_report_passed=$advance
                all_report_and_separate_drift_passed=(@($levelRows | Where-Object { -not $_.report_and_separate_drift_passed }).Count -eq 0)
                higher_speed_allowed_by_existing_report_gate=$advance
            }
            Save-Summary
        }
    }
    $summary.status = 'completed_diagnostic_only'
} catch {
    $failure = $_
    $summary.status = 'error'
    $summary.error = $_.Exception.Message
} finally {
    try {
        Assert-PreservedBaselines
        $summary.baseline_files_unchanged_after = $true
    } catch {
        $preservationFailure = $_
        $summary.status = 'error'
        $summary.baseline_files_unchanged_after = $false
        $summary.preservation_error = $_.Exception.Message
    }
    $summary.finished_at = Get-Date -Format o
    Save-Summary
}
if ($null -ne $failure -and $null -ne $preservationFailure) { throw "Probe failure: $($failure.Exception.Message). Preservation check also failed: $($preservationFailure.Exception.Message)" }
if ($null -ne $failure) { throw $failure }
if ($null -ne $preservationFailure) { throw $preservationFailure }
Write-Host "Finite probe complete: $summaryPath. No model trained, changed or promoted."
