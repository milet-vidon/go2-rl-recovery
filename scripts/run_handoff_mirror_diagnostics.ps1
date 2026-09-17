param([Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunTag,[switch]$PreflightOnly)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 3
$taskRoot='E:\IsaacLab\go2-rl-open-source'
$taskPython='E:\IsaacLab\env\python.exe'
$taskRoll='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-16_19-02-40_20260916-smithnominal128x2000\model_1999.pt'
$taskStand='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_handoff_stand\20260917-handoff-stand128x100\model_3547.pt'
$taskOutput=Join-Path $taskRoot ('evaluations\'+$RunTag)
$taskLogs=Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
$taskWrapper=Join-Path $PSScriptRoot 'evaluate_handoff_mirror.ps1'
$taskMatcher=Join-Path $PSScriptRoot 'compare_handoff_mirror_baseline.py'
if ((Test-Path -LiteralPath $taskOutput) -or (Test-Path -LiteralPath $taskLogs)) {throw 'Use a new tag; preserve existing evidence.'}
$taskSources=@($PSCommandPath,$taskWrapper,$taskMatcher,$taskRoll,$taskStand,
    (Join-Path $PSScriptRoot 'evaluate_handoff_mirror.py'),
    (Join-Path $PSScriptRoot 'evaluate_handoff_transition.py'),
    (Join-Path $PSScriptRoot 'compare_handoff_transition_baseline.py'),
    (Join-Path $PSScriptRoot 'evaluate_go2_recovery.py'),
    (Join-Path $PSScriptRoot 'verify_overnight_baselines.py'),
    (Join-Path $taskRoot 'src\go2_recovery\roll_mirror_math.py'),
    (Join-Path $taskRoot 'src\go2_recovery\handoff_transition_math.py'),
    (Join-Path $taskRoot 'src\go2_recovery\recovery_handoff_math.py'),
    (Join-Path $taskRoot 'configs\overnight_preservation_20260917.json'),
    (Join-Path $taskRoot 'datasets\recovery_states\nominal_pd_v1_20260916\states.npz'),
    (Join-Path $taskRoot 'datasets\recovery_states\nominal_pd_v1_20260916\manifest.json'),
    'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd',
    'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\Props\instanceable_meshes.usd',
    'E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd')
$taskBaselines=@{}
foreach($taskPose in @('side','upside_down','upright')) {
    $taskBaselines[$taskPose]=Join-Path $taskRoot "evaluations\20260917-handoff-transition-v1\hard-$taskPose\model_1999.pt_recovery_metrics.json"
    $taskSources+=$taskBaselines[$taskPose]
}
$taskInstalled='E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$taskSources+=@(rg --files -g '*.py' $taskInstalled)
if($LASTEXITCODE -ne 0){throw 'Could not inventory installed sources.'}
$taskHashes=@($taskSources | Sort-Object -Unique | ForEach-Object {@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash}})
function Assert-TaskInputs {
    foreach($taskSource in $taskHashes) {
        if((Get-FileHash -LiteralPath $taskSource.path -Algorithm SHA256).Hash -ne $taskSource.sha256){throw "Frozen whole-batch source drift: $($taskSource.path)"}
    }
    & $taskPython -I -B (Join-Path $PSScriptRoot 'verify_overnight_baselines.py')
    if($LASTEXITCODE -ne 0){throw 'Preserved old models changed.'}
}
Assert-TaskInputs
& $taskWrapper -Checkpoint $taskRoll -StandCheckpoint $taskStand -OutputDir (Join-Path $taskOutput 'off-side') -Mode off -Pose side -Trials 20 -Seed 20260918 -PreflightOnly
if($PreflightOnly){Write-Output 'Six finite cases planned; preflight only, no simulation or outputs.';return}
$null=New-Item -ItemType Directory -Path $taskOutput,$taskLogs
$taskStatus=@{schema='handoff_mirror_finite_batch_v1';status='running';started_at=(Get-Date).ToString('o');source_snapshot=$taskHashes;
    rows=@();acceptance_eligible=$false;promotion_performed=$false;training_collection_eligible=$false;
    limitations='Frozen two-actor inference diagnostic, reused development starts. Off must exactly match hard9/20side,20/20back,12/20upright. No change to gate/startup/PD/physics/history. No training, paper reproduction, locomotion or hardware acceptance.'}
function Save-TaskStatus {
    [IO.File]::WriteAllText((Join-Path $taskOutput 'summary.json'),($taskStatus | ConvertTo-Json -Depth 14),[Text.UTF8Encoding]::new($false))
}
Save-TaskStatus
try {
    foreach($taskMode in @('off','initial_right')) {
        # The entire off triple must finish and pass equivalence before any ON.
        foreach($taskPose in @('side','upside_down','upright')) {
            Assert-TaskInputs
            $taskCase=$taskMode+'-'+$taskPose
            $taskCaseDir=Join-Path $taskOutput $taskCase
            Write-Output ((Get-Date).ToString('o')+' Starting '+$taskCase)
            & $taskWrapper -Checkpoint $taskRoll -StandCheckpoint $taskStand -OutputDir $taskCaseDir -Mode $taskMode -Pose $taskPose -Trials 20 -Seed 20260918 *> (Join-Path $taskLogs ($taskCase+'.log'))
            $taskReportPath=Join-Path $taskCaseDir 'model_1999.pt_recovery_metrics.json'
            $taskReport=Get-Content -Raw -LiteralPath $taskReportPath | ConvertFrom-Json
            $taskComparison=$null
            if($taskMode -eq 'off') {
                $taskComparison=Join-Path $taskCaseDir 'off_equivalence.json'
                & $taskPython -B $taskMatcher --baseline $taskBaselines[$taskPose] --candidate $taskReportPath --output $taskComparison *> (Join-Path $taskLogs ($taskCase+'-equivalence.log'))
                if($LASTEXITCODE -ne 0){throw "OFF equality failed for $taskPose. Mirror ON forbidden."}
            }
            Assert-TaskInputs
            $taskStatus.rows+=@{case=$taskCase;mode=$taskMode;pose=$taskPose;trials=20;
                final_valid=$taskReport.results.$taskPose.final_valid_stands;
                selected=$taskReport.results.$taskPose.mirror_diagnostic.selected_trials;
                report=$taskReportPath;report_sha256=(Get-FileHash -LiteralPath $taskReportPath -Algorithm SHA256).Hash;
                off_equivalence=$taskComparison}
            Save-TaskStatus
        }
    }
    $taskStatus.status='completed_diagnostics_only'
} catch {
    $taskStatus.status='error';$taskStatus.error=$_.Exception.Message
    throw
} finally {
    $taskStatus.finished_at=(Get-Date).ToString('o');Save-TaskStatus
}
Write-Output 'Finite six-case mirror diagnostic completed; no promotion or automatic training.'
