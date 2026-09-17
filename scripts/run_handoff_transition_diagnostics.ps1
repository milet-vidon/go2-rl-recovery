param([Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunTag)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 3
$taskRoot='E:\IsaacLab\go2-rl-open-source'
$taskPython='E:\IsaacLab\env\python.exe'
$taskRoll='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-16_19-02-40_20260916-smithnominal128x2000\model_1999.pt'
$taskStand='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_handoff_stand\20260917-handoff-stand128x100\model_3547.pt'
$taskOutput=Join-Path $taskRoot ('evaluations\'+$RunTag)
$taskLogs=Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
if ((Test-Path -LiteralPath $taskOutput) -or (Test-Path -LiteralPath $taskLogs)) {throw 'Use a new tag; never overwrite evidence.'}
$taskPoseBaselines=@{
    upright='evaluations\20260917-handoff1999-to3547-upright-control\model_1999.pt_recovery_metrics.json';
    side='evaluations\20260917-handoff1999-to3547-side\model_1999.pt_recovery_metrics.json';
    upside_down='evaluations\20260917-handoff1999-to3547-upside_down\model_1999.pt_recovery_metrics.json'
}
foreach ($taskBase in $taskPoseBaselines.Values) {
    if (-not (Test-Path -LiteralPath (Join-Path $taskRoot $taskBase) -PathType Leaf)) {throw "Missing old baseline: $taskBase"}
}
if ((Get-FileHash -LiteralPath $taskRoll -Algorithm SHA256).Hash -ne '71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c' -or
    (Get-FileHash -LiteralPath $taskStand -Algorithm SHA256).Hash -ne '5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb') {
    throw 'This finite transition experiment requires the frozen1999/3547pair.'
}
$taskWrapper=Join-Path $PSScriptRoot 'evaluate_handoff_transition.ps1'
$taskMatcher=Join-Path $PSScriptRoot 'compare_handoff_transition_baseline.py'
& $taskWrapper -Checkpoint $taskRoll -StandCheckpoint $taskStand -OutputDir (Join-Path $taskOutput 'hard-side') -Mode hard -RampSeconds 0 -Pose side -PreflightOnly
$taskSourcePaths=@($PSCommandPath,$taskWrapper,$taskMatcher,
    (Join-Path $PSScriptRoot 'evaluate_handoff_transition.py'),
    (Join-Path $PSScriptRoot 'evaluate_go2_recovery.py'),
    (Join-Path $PSScriptRoot 'verify_overnight_baselines.py'),
    (Join-Path $taskRoot 'src\go2_recovery\handoff_transition_math.py'),
    (Join-Path $taskRoot 'src\go2_recovery\recovery_handoff_math.py'),
    $taskRoll,$taskStand,
    (Join-Path $taskRoot 'configs\overnight_preservation_20260917.json'),
    (Join-Path $taskRoot 'datasets\recovery_states\nominal_pd_v1_20260916\states.npz'),
    (Join-Path $taskRoot 'datasets\recovery_states\nominal_pd_v1_20260916\manifest.json'),
    'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd',
    'E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd')
$taskSourcePaths+=@($taskPoseBaselines.Values | ForEach-Object {Join-Path $taskRoot $_})
$taskSourcePaths+=@(rg --files -g '*.py' 'E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2')
if ($LASTEXITCODE -ne 0) {throw 'Could not inventory installed Go2 sources for whole-batch locking.'}
$taskSourceHashes=@($taskSourcePaths | ForEach-Object {@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash}})
$null=New-Item -ItemType Directory -Path $taskOutput
$null=New-Item -ItemType Directory -Path $taskLogs
$taskStatus=@{schema='handoff_transition_finite_batch_v1';status='running';started_at=(Get-Date).ToString('o');
    roll_checkpoint=$taskRoll;stand_checkpoint=$taskStand;source_snapshot=$taskSourceHashes;
    rows=@();acceptance_eligible=$false;promotion_performed=$false;
    limitations='Development diagnosis only; frozen two-policy actors. Original dual upright12/side9/back20, not stand-only20. No learned selector, no retry, no locomotion or hardware acceptance.'}
function Save-TaskStatus {
    [IO.File]::WriteAllText((Join-Path $taskOutput 'summary.json'),($taskStatus | ConvertTo-Json -Depth 14),[Text.UTF8Encoding]::new($false))
}
function Assert-TaskSources {
    foreach ($taskSource in $taskSourceHashes) {
        if ((Get-FileHash -LiteralPath $taskSource.path -Algorithm SHA256).Hash -ne $taskSource.sha256) {
            throw "Pinned batch source changed: $($taskSource.path)"
        }
    }
}
Save-TaskStatus
try {
    # All THREE hard controls must match the original results before any ramp.
    foreach ($taskPose in @('side','upside_down','upright')) {
        Assert-TaskSources
        $taskCase='hard-'+$taskPose
        $taskCaseDir=Join-Path $taskOutput $taskCase
        Write-Output ((Get-Date).ToString('o')+' Starting '+$taskCase)
        & $taskWrapper -Checkpoint $taskRoll -StandCheckpoint $taskStand -OutputDir $taskCaseDir -Mode hard -RampSeconds 0 -Pose $taskPose -Trials 20 -Seed 20260918 *> (Join-Path $taskLogs ($taskCase+'.log'))
        $taskReportPath=Join-Path $taskCaseDir 'model_1999.pt_recovery_metrics.json'
        $taskComparison=Join-Path $taskCaseDir 'hard_equivalence.json'
        & $taskPython -B $taskMatcher --baseline (Join-Path $taskRoot $taskPoseBaselines[$taskPose]) --candidate $taskReportPath --output $taskComparison *> (Join-Path $taskLogs ($taskCase+'-equivalence.log'))
        if ($LASTEXITCODE -ne 0) {throw "Hard-equivalence failed for $taskPose; ramps are forbidden."}
        $taskReport=Get-Content -Raw -LiteralPath $taskReportPath | ConvertFrom-Json
        $taskStatus.rows+=@{case=$taskCase;mode='hard';ramp_seconds=0;pose=$taskPose;trials=20;
            final_valid=$taskReport.results.$taskPose.final_valid_stands;report=$taskReportPath;
            report_sha256=(Get-FileHash -LiteralPath $taskReportPath -Algorithm SHA256).Hash;hard_equivalence=$taskComparison}
        Save-TaskStatus
    }
    foreach ($taskRamp in @(0.2,0.5)) {
        foreach ($taskPose in @('side','upside_down','upright')) {
            Assert-TaskSources
            $taskRampTag=if ($taskRamp -eq 0.2) {'raw02'} else {'raw05'}
            $taskCase=$taskRampTag+'-'+$taskPose
            $taskCaseDir=Join-Path $taskOutput $taskCase
            Write-Output ((Get-Date).ToString('o')+' Starting '+$taskCase)
            & $taskWrapper -Checkpoint $taskRoll -StandCheckpoint $taskStand -OutputDir $taskCaseDir -Mode raw_ramp -RampSeconds $taskRamp -Pose $taskPose -Trials 20 -Seed 20260918 *> (Join-Path $taskLogs ($taskCase+'.log'))
            $taskReportPath=Join-Path $taskCaseDir 'model_1999.pt_recovery_metrics.json'
            $taskReport=Get-Content -Raw -LiteralPath $taskReportPath | ConvertFrom-Json
            $taskStatus.rows+=@{case=$taskCase;mode='raw_ramp';ramp_seconds=$taskRamp;pose=$taskPose;trials=20;
                final_valid=$taskReport.results.$taskPose.final_valid_stands;
                geometry_pass=$taskReport.results.$taskPose.final_geometry_passes;
                report=$taskReportPath;report_sha256=(Get-FileHash -LiteralPath $taskReportPath -Algorithm SHA256).Hash}
            Save-TaskStatus
        }
    }
    Assert-TaskSources
    $taskStatus.status='completed_diagnostics_only'
    $taskStatus.finished_at=(Get-Date).ToString('o')
    Save-TaskStatus
    Write-Output 'Finite9-case transition diagnostics completed. No promotion or automatic training.'
} catch {
    $taskStatus.status='failed'
    $taskStatus.error=$_.Exception.Message
    $taskStatus.finished_at=(Get-Date).ToString('o')
    Save-TaskStatus
    throw
}
