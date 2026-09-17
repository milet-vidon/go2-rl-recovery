param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunTag,
    [switch]$ExecuteFiniteBatch,
    [switch]$PreflightOnly
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 3
$taskRoot='E:\IsaacLab\go2-rl-open-source'
$taskPython='E:\IsaacLab\env\python.exe'
$taskRoll='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-16_19-02-40_20260916-smithnominal128x2000\model_1999.pt'
$taskStand='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_handoff_stand\20260917-handoff-stand128x100\model_3547.pt'
$taskOutput=Join-Path $taskRoot ('evaluations\'+$RunTag)
$taskLogs=Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
$taskWrapper=Join-Path $PSScriptRoot 'evaluate_handoff_supported_startup.ps1'
$taskMatcher=Join-Path $PSScriptRoot 'compare_handoff_supported_startup.py'
$taskStandReference=Join-Path $taskRoot 'evaluations\20260917-handoff-stand100-upright20\model_3547.pt_recovery_metrics.json'
if((Test-Path -LiteralPath $taskOutput) -or (Test-Path -LiteralPath $taskLogs)){throw 'Use a new RunTag; never overwrite old evidence.'}
function Assert-TaskIdle {
    $taskBusy=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit.*|isaac-sim.*)\.exe$' -and
        ($_.CommandLine -match 'E:[\\/]IsaacLab' -or $_.ExecutablePath -match '^E:[\\/]IsaacLab[\\/]')
    })
    if($taskBusy.Count -gt 0){throw "Workspace Python/Kit already active: $($taskBusy.ProcessId -join ',')"}
}
Assert-TaskIdle
$taskSources=@($PSCommandPath,$taskWrapper,$taskMatcher,$taskRoll,$taskStand,$taskStandReference,
    (Join-Path $PSScriptRoot 'evaluate_handoff_supported_startup.py'),
    (Join-Path $PSScriptRoot 'test_handoff_supported_startup_entry.py'),
    (Join-Path $PSScriptRoot 'test_supported_startup_math.py'),
    (Join-Path $PSScriptRoot 'evaluate_handoff_transition.py'),
    (Join-Path $PSScriptRoot 'compare_handoff_transition_baseline.py'),
    (Join-Path $PSScriptRoot 'evaluate_go2_recovery.py'),
    (Join-Path $PSScriptRoot 'verify_overnight_baselines.py'),
    (Join-Path $taskRoot 'src\go2_recovery\supported_startup_math.py'),
    (Join-Path $taskRoot 'src\go2_recovery\handoff_transition_math.py'),
    (Join-Path $taskRoot 'src\go2_recovery\recovery_handoff_math.py'),
    (Join-Path $taskRoot 'configs\overnight_preservation_20260917.json'),
    (Join-Path $taskRoot 'docs\handoff-supported-startup-preregistration-20260917.md'),
    (Join-Path $taskRoot 'datasets\recovery_states\nominal_pd_v1_20260916\states.npz'),
    (Join-Path $taskRoot 'datasets\recovery_states\nominal_pd_v1_20260916\manifest.json'),
    'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd',
    'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\Props\instanceable_meshes.usd',
    'E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd')
$taskBaselines=@{}
foreach($taskPose in @('upright','side','upside_down')) {
    $taskBaseDir=Join-Path $taskRoot "evaluations\20260917-handoff-transition-v1\hard-$taskPose"
    $taskBaselines[$taskPose]=Join-Path $taskBaseDir 'model_1999.pt_recovery_metrics.json'
    $taskSources+=@($taskBaselines[$taskPose],
        (Join-Path $taskBaseDir 'model_1999.pt_recovery_trace.csv'),
        (Join-Path $taskBaseDir 'model_1999.pt_handoff_transition_trace.json'))
}
$taskSources+=Join-Path (Split-Path $taskStandReference -Parent) 'model_3547.pt_recovery_trace.csv'
$taskInstalled='E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$taskSources+=@(rg --files -g '*.py' $taskInstalled)
if($LASTEXITCODE -ne 0){throw 'Could not inventory installed source.'}
$taskHashes=@($taskSources | Sort-Object -Unique | ForEach-Object {
    @{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
})
function Assert-TaskInputs {
    foreach($taskSource in $taskHashes){
        if((Get-FileHash -LiteralPath $taskSource.path -Algorithm SHA256).Hash -ne $taskSource.sha256){throw "Frozen batch source drift: $($taskSource.path)"}
    }
    Assert-TaskIdle
    & $taskPython -I -B (Join-Path $PSScriptRoot 'verify_overnight_baselines.py')
    if($LASTEXITCODE -ne 0){throw 'Old model preservation failed.'}
}
Assert-TaskInputs
& $taskPython -B (Join-Path $PSScriptRoot 'test_supported_startup_math.py')
if($LASTEXITCODE -ne 0){throw 'Pure startup predicate regression failed.'}
& $taskPython -B (Join-Path $PSScriptRoot 'test_handoff_supported_startup_entry.py')
if($LASTEXITCODE -ne 0){throw 'Startup entry/recording CPU regression failed.'}
& $taskWrapper -Checkpoint $taskRoll -StandCheckpoint $taskStand -OutputDir (Join-Path $taskOutput 'off-upright') -Mode off -Pose upright -Trials 20 -Seed 20260918 -PreflightOnly
if($PreflightOnly -or -not $ExecuteFiniteBatch){Write-Output 'PREFLIGHT ONLY: finite off3 exact hard controls, then supported3 matched comparisons. Use -ExecuteFiniteBatch to run.';return}
$null=New-Item -ItemType Directory -Path $taskOutput,$taskLogs
$taskStatus=@{schema='handoff_supported_startup_finite_batch_v1';status='running';started_at=(Get-Date).ToString('o');
    source_snapshot=$taskHashes;rows=@();acceptance_eligible=$false;promotion_performed=$false;
    training_performed=$false;training_collection_eligible=$false;
    limitations='Development-only two-actor startup routing, no mirror/ramp/training. Off all three must exactly match hard controls before on. Selected upright compares stand-only20/20 actual starts/outcomes/trial0 CSV; unselected side/back exact hard retention. No generalization, integrated locomotion or hardware claim.'}
function Save-TaskStatus {
    [IO.File]::WriteAllText((Join-Path $taskOutput 'summary.json'),($taskStatus | ConvertTo-Json -Depth 16),[Text.UTF8Encoding]::new($false))
}
Save-TaskStatus
try {
    foreach($taskMode in @('off','supported')) {
        foreach($taskPose in @('upright','side','upside_down')) {
            Assert-TaskInputs
            $taskCase=$taskMode+'-'+$taskPose
            $taskCaseDir=Join-Path $taskOutput $taskCase
            Write-Output ((Get-Date).ToString('o')+' Starting '+$taskCase)
            & $taskWrapper -Checkpoint $taskRoll -StandCheckpoint $taskStand -OutputDir $taskCaseDir -Mode $taskMode -Pose $taskPose -Trials 20 -Seed 20260918 *> (Join-Path $taskLogs ($taskCase+'.log'))
            $taskReportPath=Join-Path $taskCaseDir 'model_1999.pt_recovery_metrics.json'
            $taskReport=Get-Content -Raw -LiteralPath $taskReportPath | ConvertFrom-Json
            $taskBaseline=if($taskMode -eq 'supported' -and $taskPose -eq 'upright'){$taskStandReference}else{$taskBaselines[$taskPose]}
            $taskComparison=Join-Path $taskCaseDir 'exact_comparison.json'
            & $taskPython -B $taskMatcher --baseline $taskBaseline --candidate $taskReportPath --output $taskComparison *> (Join-Path $taskLogs ($taskCase+'-comparison.log'))
            $taskComparisonExit=$LASTEXITCODE
            $taskStatus.rows+=@{case=$taskCase;mode=$taskMode;pose=$taskPose;trials=20;
                final_valid=$taskReport.results.$taskPose.final_valid_stands;
                selected=$taskReport.results.$taskPose.startup_selection.selected_trials;
                genuine_handoffs=$taskReport.results.$taskPose.handoff_diagnostic.triggered_trials;
                comparison_passed=($taskComparisonExit -eq 0);comparison=$taskComparison;
                report=$taskReportPath;report_sha256=(Get-FileHash -LiteralPath $taskReportPath -Algorithm SHA256).Hash.ToLowerInvariant()}
            Save-TaskStatus
            if($taskComparisonExit -ne 0){throw "Exact preregistered comparison failed: $taskCase. No automatic continuation, threshold change or promotion."}
            Assert-TaskInputs
        }
    }
    $taskStatus.status='completed_diagnostics_only'
} catch {
    $taskStatus.status='error';$taskStatus.error=$_.Exception.Message
    throw
} finally {
    $taskStatus.finished_at=(Get-Date).ToString('o');Save-TaskStatus
}
Write-Output 'Finite startup diagnostic completed. No training, mirror combination or model promotion.'
