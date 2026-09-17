param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$StandCheckpoint,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [Parameter(Mandatory=$true)][ValidateSet('off','supported')][string]$RefinementMode,
    [Parameter(Mandatory=$true)][ValidateSet('off','supported')][string]$Mode,
    [Parameter(Mandatory=$true)][ValidateSet('off','initial_right')][string]$MirrorMode,
    [ValidateSet('upright','side','upside_down')][string]$Pose = 'side',
    [ValidateSet(1,2,20)][int]$Trials = 20,
    [int]$Seed = 20260918,
    [ValidateSet('oblique','front','side')][string]$View = 'oblique',
    [switch]$Video,
    [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3
$taskRoot = 'E:\IsaacLab\go2-rl-open-source'
$taskWorkspace = 'E:\IsaacLab\'
$taskPython = 'E:\IsaacLab\env\python.exe'
$taskName = 'Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0'
$taskBank = Join-Path $taskRoot 'datasets\recovery_states\nominal_pd_v1_20260916'
$taskEntry = Join-Path $PSScriptRoot 'evaluate_supported_refinement.py'
$taskOldEntry = Join-Path $PSScriptRoot 'evaluate_go2_recovery.py'
$taskOldSha = '4024ba713e3ed616225d64e1d59de97fd31c8369bba83ac3f4235d93552e9e2f'
foreach ($taskInput in @($Checkpoint,$StandCheckpoint,$taskEntry,$taskOldEntry)) {
    if (-not (Test-Path -LiteralPath $taskInput -PathType Leaf)) {throw "Missing input: $taskInput"}
    if (-not [IO.Path]::GetFullPath($taskInput).StartsWith($taskWorkspace,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'All experimental inputs must remain under E:\IsaacLab.'
    }
}
$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
$StandCheckpoint = (Resolve-Path -LiteralPath $StandCheckpoint).Path
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if (-not $OutputDir.StartsWith((Join-Path $taskRoot 'evaluations\'),[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Use a new directory below E:\IsaacLab\go2-rl-open-source\evaluations.'
}
if (Test-Path -LiteralPath $OutputDir) {throw 'Output already exists; preserve old evidence and choose a new name.'}
if ((Get-FileHash -LiteralPath $taskOldEntry -Algorithm SHA256).Hash -ne $taskOldSha) {
    throw 'RAW-pinned original evaluator changed; do not silently reuse baseline provenance.'
}
if ((Get-FileHash -LiteralPath (Join-Path $taskBank 'states.npz') -Algorithm SHA256).Hash -ne '71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a') {
    throw 'Frozen original state bank changed.'
}
function Assert-TransitionIdle {
    $taskBusy = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit.*|isaac-sim.*)\.exe$' -and
        ($_.CommandLine -match 'E:[\\/]IsaacLab' -or $_.ExecutablePath -match '^E:[\\/]IsaacLab[\\/]')
    })
    if ($taskBusy.Count -gt 0) {throw "E-drive Python/Kit process already active: $($taskBusy.ProcessId -join ','). No parallel simulator."}
}
Assert-TransitionIdle
$taskAngle = if ($Pose -eq 'upright') {'0'} else {'30'}
$taskArgs = @('-B',$taskEntry,'--task',$taskName,'--checkpoint',$Checkpoint,
    '--refinement_mode',$RefinementMode,'--stand_checkpoint',$StandCheckpoint,'--output_dir',$OutputDir,'--startup_mode',$Mode,'--mirror_mode',$MirrorMode,
    '--trials',"$Trials",'--seed',"$Seed",'--poses',$Pose,
    '--angle_deg',$taskAngle,'--settle_s','1','--horizon_s','8','--hold_s','3',
    '--min_contacts','4','--device','cuda:0','--headless','--kit_args=--/app/vulkan=false')
if ($Pose -ne 'upright') {$taskArgs += @('--state_bank_path',$taskBank,'--state_bank_split','heldout')}
if ($Video) {$taskArgs += @('--video_pose','all','--view',$View)}
& $taskPython -B (Join-Path $PSScriptRoot 'verify_overnight_baselines.py')
if ($LASTEXITCODE -ne 0) {throw 'Frozen model preservation failed before experiment.'}
$taskPaths = @($PSCommandPath,$taskEntry,$taskOldEntry,$Checkpoint,$StandCheckpoint,
    (Join-Path $PSScriptRoot 'supported_refinement_gate.py'),
    (Join-Path $PSScriptRoot 'evaluate_natural_handoff_candidate.py'),
    (Join-Path $PSScriptRoot 'evaluate_handoff_combined.py'),
    'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_handoff_stand\20260917-natural-stand128x200-first\training_result.json',
    'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_handoff_stand\20260917-natural-stand128x200-first\model_3746.pt',
    (Join-Path $PSScriptRoot 'evaluate_handoff_transition.py'),
    (Join-Path $PSScriptRoot 'evaluate_handoff_mirror.py'),
    (Join-Path $taskRoot 'src\go2_recovery\roll_mirror_math.py'),
    (Join-Path $taskRoot 'src\go2_recovery\supported_startup_math.py'),
    (Join-Path $taskRoot 'src\go2_recovery\handoff_transition_math.py'),
    (Join-Path $taskRoot 'src\go2_recovery\recovery_handoff_math.py'),
    (Join-Path $taskBank 'states.npz'),(Join-Path $taskBank 'manifest.json'),
    (Join-Path $PSScriptRoot 'verify_overnight_baselines.py'),
    'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd',
    'E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd')
$taskInstalled = 'E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$taskPaths += @(rg --files -g '*.py' $taskInstalled)
if ($LASTEXITCODE -ne 0) {throw 'Could not inventory installed Go2 sources.'}
$taskSnapshot = @($taskPaths | Sort-Object -Unique | ForEach-Object {
    if (-not (Test-Path -LiteralPath $_ -PathType Leaf)) {throw "Missing provenance file: $_"}
    @{path=[IO.Path]::GetFullPath($_);sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
})
if ($PreflightOnly) {
    Write-Output ('PREFLIGHT ONLY (dependencies/inputs checked, no simulation): ' + $taskPython + ' ' + ($taskArgs -join ' '))
    return
}
$null = New-Item -ItemType Directory -Path $OutputDir
$taskInvocation = @{schema='supported_refinement_invocation_v1';started_at=(Get-Date).ToString('o');
    mode=$Mode;mirror_mode=$MirrorMode;refinement_mode=$RefinementMode;ramp_seconds=0;pose=$Pose;trials=$Trials;seed=$Seed;args=$taskArgs;
    source_and_inputs=$taskSnapshot;notice='SIMULATION DIAGNOSTIC ONLY; no training, promotion or hardware acceptance.'}
[IO.File]::WriteAllText((Join-Path $OutputDir 'invocation.json'),($taskInvocation | ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
$env:OMNI_KIT_ACCEPT_EULA='YES'
$env:OMNI_USER_HOME='E:\IsaacLab\userdata'
$env:OV_USER_HOME='E:\IsaacLab\userdata'
$env:TEMP='E:\IsaacLab\tmp'
$env:TMP='E:\IsaacLab\tmp'
$env:PIP_CACHE_DIR='E:\IsaacLab\cache\pip'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:ISAACLAB_GO2_USD='E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd'
$env:ISAACLAB_GROUND_USD='E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd'
$env:CONDA_PREFIX='E:\IsaacLab\env'
$env:Path='E:\IsaacLab\env;E:\IsaacLab\env\Scripts;'+$env:Path
$taskPreviousBank=[Environment]::GetEnvironmentVariable('ISAACLAB_RECOVERY_BANK_COLLECTION','Process')
try {
    $env:ISAACLAB_RECOVERY_BANK_COLLECTION='1'
    Assert-TransitionIdle
    $taskSimLog=Join-Path $OutputDir 'simulator.log'
    & $taskPython @taskArgs *> $taskSimLog
    if ($LASTEXITCODE -ne 0) {throw "Transition evaluation exited with $LASTEXITCODE"}
} finally {
    if ($null -eq $taskPreviousBank) {Remove-Item Env:ISAACLAB_RECOVERY_BANK_COLLECTION -ErrorAction SilentlyContinue}
    else {$env:ISAACLAB_RECOVERY_BANK_COLLECTION=$taskPreviousBank}
}
$taskFatal=@(rg -n -i 'CUDA error|CUDA_ERROR|out of memory|PhysX.*(error|overflow)|Pxg.*(error|overflow)|GPU.*buffer.*overflow|Traceback \(most recent call last\)|Fatal Python error|illegal memory access|\[Error\].*omni\.physx' $taskSimLog)
if ($LASTEXITCODE -notin @(0,1)) {throw 'Could not audit simulator log.'}
if ($taskFatal.Count -gt 0) {throw ('Simulation error evidence: '+($taskFatal -join ' | '))}
foreach ($taskItem in $taskSnapshot) {
    if ((Get-FileHash -LiteralPath $taskItem.path -Algorithm SHA256).Hash -ne $taskItem.sha256) {
        throw "Source or input changed during experiment: $($taskItem.path)"
    }
}
$taskReportPath=Join-Path $OutputDir ((Split-Path $Checkpoint -Leaf)+'_recovery_metrics.json')
$taskReport=Get-Content -Raw -LiteralPath $taskReportPath | ConvertFrom-Json
if ($taskReport.protocol_version -ne 'supported_postrecovery_refinement_v1' -or
    $taskReport.task -ne $taskName -or $taskReport.seed -ne $Seed -or
    $taskReport.settle_requested_s -ne 1 -or $taskReport.settle_actual_s -ne 1 -or
    $taskReport.settle_control_steps -ne 50 -or $taskReport.policy_action_mode -ne 'deterministic_mean' -or
    $taskReport.startup_experiment.mode -ne $Mode -or
    $taskReport.mirror_experiment.mode -ne $MirrorMode -or
    $taskReport.startup_experiment.mirror_enabled -ne ($MirrorMode -eq 'initial_right') -or
    $taskReport.startup_experiment.selected_counts_as_handoff -ne $false -or
    $taskReport.startup_experiment.physical_state_or_history_mutated -ne $false -or
    $taskReport.transition_experiment.mode -ne 'hard' -or
    $taskReport.transition_experiment.ramp_seconds -ne 0 -or
    $taskReport.transition_experiment.step_dt -ne 0.02 -or
    $taskReport.transition_experiment.horizon_s -ne 8 -or
    $taskReport.transition_experiment.hold_s -ne 3 -or
    $taskReport.transition_experiment.min_contacts -ne 4 -or
    $taskReport.acceptance_eligible -ne $false -or
    $taskReport.single_policy_acceptance_eligible -ne $false -or
    $taskReport.training_collection_eligible -ne $false -or
    $taskReport.results.$Pose.trials -ne $Trials) {throw 'Experimental report identity/denominator mismatch.'}
if (@($taskReport.results.PSObject.Properties).Count -ne 1) {throw 'Exactly one pose per cold process is required.'}
$taskRefinement=$taskReport.refinement_experiment
if ($taskRefinement.mode -ne $RefinementMode -or $taskRefinement.num_envs -ne $Trials -or
    $taskRefinement.qualifying_completed_intervals -ne 150 -or $taskRefinement.step_dt -ne .02 -or
    $taskRefinement.decision_applies_to_next_action -ne $true -or
    $taskRefinement.switch_time_physical_or_history_reset -ne $false -or
    $taskRefinement.extra_pd_ramp_or_retry -ne $false -or
    $taskRefinement.poses.$Pose.intervals -ne 550 -or $taskRefinement.full_measured_rows -ne (550*$Trials) -or
    $taskRefinement.refinement_actor_training.checkpoint_sha256 -ne 'f40772c228996400a9813ad244b48509737dabf2c31af001144de34283433b9e') {throw 'Refinement receipt/protocol mismatch.'}
$taskFullTrace=Join-Path $OutputDir 'supported_refinement_full_trace.jsonl'
if ($taskRefinement.trace_file -cne 'supported_refinement_full_trace.jsonl' -or
    (Get-FileHash -LiteralPath $taskFullTrace).Hash -ne $taskRefinement.trace_sha256) {throw 'Full measured trace SHA mismatch.'}
$taskPoseResult=$taskReport.results.$Pose
foreach ($taskCountName in @('successes','final_valid_stands','final_geometry_passes','settled_fallen_trials')) {
    $taskCount=$taskPoseResult.$taskCountName
    if (($taskCount -isnot [int] -and $taskCount -isnot [long]) -or $taskCount -lt 0 -or $taskCount -gt $Trials) {
        throw "Invalid result count: $taskCountName"
    }
}
if ($taskPoseResult.final_valid_stands -gt $taskPoseResult.successes -or
    $taskPoseResult.final_valid_stands -gt $taskPoseResult.final_geometry_passes) {throw 'Contradictory success/geometry counts.'}
foreach ($taskListName in @('final_diagnostics','policy_start_state','release_state_before_settling')) {
    if (@($taskPoseResult.$taskListName).Count -ne $Trials) {throw "Incomplete per-trial records: $taskListName"}
}
for ($taskTrial=0;$taskTrial -lt $Trials;$taskTrial++) {
    if ($taskPoseResult.final_diagnostics[$taskTrial].trial -ne $taskTrial) {throw 'Final trial ordering mismatch.'}
}
if ($Pose -ne 'upright' -and ($taskReport.state_bank.split -ne 'heldout' -or
    $taskReport.state_bank.sha256 -ne '71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a' -or
    @($taskReport.state_bank.selected_state_ids).Count -ne $Trials)) {throw 'Report bank provenance mismatch.'}
if ($taskReport.checkpoint_sha256 -ne (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash -or
    $taskReport.handoff_controller.stand_sha256 -ne (Get-FileHash -LiteralPath $StandCheckpoint -Algorithm SHA256).Hash) {
    throw 'Report checkpoint identity mismatch.'
}
$taskExpectedTrace=(Split-Path $Checkpoint -Leaf)+'_handoff_combined_trace.json'
if ($taskReport.transition_experiment.transition_trace_file -cne $taskExpectedTrace) {throw 'Trace must belong to this case directory.'}
$taskTrace = Join-Path $OutputDir $taskExpectedTrace
if (-not (Test-Path -LiteralPath $taskTrace -PathType Leaf)) {throw 'Missing all-trial transition trace.'}
if ((Get-FileHash -LiteralPath $taskTrace -Algorithm SHA256).Hash -ne $taskReport.transition_experiment.transition_trace_sha256) {
    throw 'Transition trace SHA mismatch.'
}
$taskTraceReport=Get-Content -Raw -LiteralPath $taskTrace | ConvertFrom-Json
if ($taskTraceReport.schema -ne 'handoff_combined_neighborhood_v1' -or
    @($taskTraceReport.poses.PSObject.Properties).Count -ne 1 -or
    @($taskTraceReport.poses.$Pose).Count -ne $Trials) {throw 'Incomplete all-trial trace.'}
for ($taskTrial=0;$taskTrial -lt $Trials;$taskTrial++) {
    if ($taskTraceReport.poses.$Pose[$taskTrial].trial -ne $taskTrial -or
        @($taskTraceReport.poses.$Pose[$taskTrial].rows).Count -eq 0) {throw 'Missing or reordered trial trace.'}
}
& $taskPython -B (Join-Path $PSScriptRoot 'verify_overnight_baselines.py')
if ($LASTEXITCODE -ne 0) {throw 'Frozen model preservation failed after experiment.'}
if (@($taskTraceReport.first_action_by_pose.$Pose).Count -ne $Trials) {throw 'Incomplete first-action trace.'}
if (@($taskPoseResult.startup_selection.selection_records).Count -ne $Trials -or
    $taskPoseResult.startup_selection.selected_genuine_handoffs -ne 0) {throw 'Invalid startup/handoff separation.'}
$taskGenerated=Join-Path $OutputDir 'combined_generated.py'
if((Get-FileHash -LiteralPath $taskGenerated -Algorithm SHA256).Hash -ne $taskReport.startup_experiment.generated_source_sha256){throw 'Generated source SHA mismatch.'}
for($taskTrial=0;$taskTrial -lt $Trials;$taskTrial++) {
    $taskRecord=$taskPoseResult.startup_selection.selection_records[$taskTrial]
    $taskFirst=$taskTraceReport.first_action_by_pose.$Pose[$taskTrial]
    if($taskRecord.trial -ne $taskTrial -or $taskRecord.valid -ne $true -or
       $taskFirst.trial -ne $taskTrial -or $taskFirst.step -ne 0 -or
       $taskFirst.startup_selected -ne $taskRecord.selected -or
       $taskFirst.just_switched -ne $false -or $taskFirst.switched -ne $false -or
       $taskFirst.issued_action_target_history_assertions_passed -ne $true){throw 'First-action/startup provenance mismatch.'}
}
Write-Output "REFINEMENT DIAGNOSTIC COMPLETE refinement=$RefinementMode startup=$Mode mirror=$MirrorMode pose=$Pose final=$($taskReport.results.$Pose.final_valid_stands)/$Trials. Not acceptance or promotion."
