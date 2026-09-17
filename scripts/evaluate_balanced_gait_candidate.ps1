param(
    [Parameter(Mandatory=$true)][ValidateSet('control','balanced')][string]$Arm,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$TrainingResult,
    [Parameter(Mandatory=$true)][ValidateSet('original','recovery')][string]$PhysicsMode,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [double]$WalkSpeed = 0.5,
    [double]$LateralSpeed = 0,
    [double]$YawRate = 0,
    [double]$PushSpeed = 0,
    [int]$Seed = 20260909,
    [ValidateSet('legacy','front','oblique')][string]$View = 'legacy',
    [switch]$NoVideo,
    [switch]$PreflightOnly
)
# Verified formal candidate diagnostic; unchanged actual retention physics/metrics. No promotion.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3
$taskRoot = 'E:\IsaacLab\go2-rl-open-source'
$taskPython = 'E:\IsaacLab\env\python.exe'
$taskRepo = 'E:\IsaacLab\repo'
$taskName = 'Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0'
$taskCheckpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
$TrainingResult = (Resolve-Path -LiteralPath $TrainingResult).Path
if (-not $taskCheckpoint.StartsWith('E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_flat\',[StringComparison]::OrdinalIgnoreCase) -or (Split-Path $taskCheckpoint -Leaf) -cne 'model_4246.pt') {throw 'Only isolated formal candidate4246.'}
$taskCheckpointSha = (Get-FileHash -LiteralPath $taskCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
$taskEntry = Join-Path $PSScriptRoot 'evaluate_balanced_gait_candidate.py'
$taskBaseEntry = Join-Path $PSScriptRoot 'evaluate_locomotion_recovery_physics.py'
if ((Get-FileHash -LiteralPath $taskBaseEntry).Hash -ne 'cc2437ffb0d84da97eda5906333e54070f2d065ea475a148b6be7f1c6cfea4c8') {throw 'Frozen physical adapter changed.'}
$taskTemplate = Join-Path $PSScriptRoot 'evaluate_go2_stand_walk_stop.py'
$taskTemplateSha = 'bbba369177d95bc24163bdd2b5ba96506f5db7d4b1ce38875f8a8885c49e22e4'
$taskVerifier = Join-Path $PSScriptRoot 'verify_overnight_baselines.py'
$taskInstalled = Join-Path $taskRepo 'source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if (-not $OutputDir.StartsWith((Join-Path $taskRoot 'evaluations\'),[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Use a new directory below E:\IsaacLab\go2-rl-open-source\evaluations.'
}
if (Test-Path -LiteralPath $OutputDir) {throw 'Output exists; preserve evidence and choose a new name.'}
foreach ($taskValue in @($WalkSpeed,$LateralSpeed,$YawRate,$PushSpeed)) {
    if ([double]::IsNaN($taskValue) -or [double]::IsInfinity($taskValue)) {throw 'Commands/push must be finite.'}
}
foreach ($taskInput in @($taskPython,$taskCheckpoint,$taskEntry,$taskTemplate,$taskVerifier)) {
    if (-not (Test-Path -LiteralPath $taskInput -PathType Leaf)) {throw "Missing input: $taskInput"}
}
if ((Get-FileHash -LiteralPath $taskCheckpoint -Algorithm SHA256).Hash -ne $taskCheckpointSha) {throw 'Candidate checkpoint changed.'}
if ((Get-FileHash -LiteralPath $taskTemplate -Algorithm SHA256).Hash -ne $taskTemplateSha) {throw 'Original metric evaluator changed.'}
function Assert-RetentionIdle {
    # Conservative: refuse all E-workspace Python/Kit, including unfamiliar entries.
    # Do not inspect, stop, or interfere with unrelated F-drive workloads.
    $taskBusy = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit.*|isaac-sim.*)\.exe$' -and
        ($_.CommandLine -match 'E:[\\/]IsaacLab' -or $_.ExecutablePath -match '^E:[\\/]IsaacLab[\\/]')
    })
    if ($taskBusy.Count -gt 0) {throw "E-drive Python/Kit already active: $($taskBusy.ProcessId -join ','); no parallel simulator."}
}
Assert-RetentionIdle
& $taskPython -B $taskEntry --arm $Arm --candidate_training_result $TrainingResult --checkpoint $taskCheckpoint --verify-only
if ($LASTEXITCODE -ne 0) {throw 'Actual candidate formal training verification failed.'}
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:OMNI_USER_HOME = 'E:\IsaacLab\userdata'
$env:OV_USER_HOME = 'E:\IsaacLab\userdata'
$env:TEMP = 'E:\IsaacLab\tmp'
$env:TMP = 'E:\IsaacLab\tmp'
$env:PIP_CACHE_DIR = 'E:\IsaacLab\cache\pip'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:ISAACLAB_GO2_USD = 'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd'
$env:ISAACLAB_GROUND_USD = 'E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd'
$env:CONDA_PREFIX = 'E:\IsaacLab\env'
$env:Path = 'E:\IsaacLab\env;E:\IsaacLab\env\Scripts;' + $env:Path
$taskPaths = @($PSCommandPath,$taskEntry,$taskTemplate,$taskVerifier,$taskCheckpoint,$TrainingResult,$taskBaseEntry,
    (Join-Path $PSScriptRoot 'train_balanced_gait_adaptation.py'),
    (Join-Path $PSScriptRoot 'test_balanced_gait_candidate.py'),
    (Join-Path $PSScriptRoot 'evaluate_common_physics_candidate.py'),
    (Join-Path $PSScriptRoot 'balanced_gait_reward.py'),
    (Join-Path $PSScriptRoot 'balanced_gait_exposure.py'),
    (Join-Path (Split-Path $TrainingResult) 'common_physics_pretrain_guard.json'),
    (Join-Path (Split-Path $TrainingResult) 'common_physics_invocation.json'),
    (Join-Path (Split-Path $TrainingResult) 'params\env.yaml'),
    (Join-Path (Split-Path $TrainingResult) 'params\agent.yaml'),
    (Join-Path $PSScriptRoot 'test_locomotion_recovery_physics.py'),
    (Join-Path $PSScriptRoot 'evaluate_stand_walk_stop.ps1'),
    (Join-Path $PSScriptRoot 'evaluate_speed_candidate.ps1'),
    $env:ISAACLAB_GO2_USD,$env:ISAACLAB_GROUND_USD,
    (Join-Path $taskRepo 'source\isaaclab_assets\isaaclab_assets\robots\unitree.py'),
    (Join-Path $taskRepo 'source\isaaclab\isaaclab\envs\mdp\actions\joint_actions.py'),
    (Join-Path $taskRepo 'source\isaaclab\isaaclab\managers\action_manager.py'),
    (Join-Path $taskRepo 'source\isaaclab\isaaclab\envs\manager_based_rl_env.py'),
    (Join-Path $taskRepo 'source\isaaclab_rl\isaaclab_rl\rsl_rl\vecenv_wrapper.py'))
foreach ($taskSourceRoot in @($taskInstalled,(Join-Path $taskRoot 'src\go2_recovery'))) {
    $taskPaths += @(rg --files -g '*.py' $taskSourceRoot)
    if ($LASTEXITCODE -ne 0) {throw "Could not inventory Python sources: $taskSourceRoot"}
}
$taskSnapshot = @($taskPaths | Sort-Object -Unique | ForEach-Object {
    if (-not (Test-Path -LiteralPath $_ -PathType Leaf)) {throw "Missing provenance file: $_"}
    @{path=[IO.Path]::GetFullPath($_);sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
})
function Assert-RetentionIntegrity {
    foreach ($taskItem in $taskSnapshot) {
        if ((Get-FileHash -LiteralPath $taskItem.path -Algorithm SHA256).Hash -ne $taskItem.sha256) {
            throw "Source/input changed during experiment: $($taskItem.path)"
        }
    }
    & $taskPython -B $taskVerifier
    if ($LASTEXITCODE -ne 0) {throw 'Frozen portfolio models changed.'}
    & $taskPython -B $taskEntry --arm $Arm --candidate_training_result $TrainingResult --checkpoint $taskCheckpoint --verify-only
    if ($LASTEXITCODE -ne 0) {throw 'Full candidate training/smoke/source proof changed during evaluation.'}
}
Assert-RetentionIntegrity
$taskCulture = [Globalization.CultureInfo]::InvariantCulture
$taskArgs = @('-B',$taskEntry,'--arm',$Arm,'--candidate_training_result',$TrainingResult,'--physics_mode',$PhysicsMode,'--task',$taskName,'--checkpoint',$taskCheckpoint,
    '--output_dir',$OutputDir,'--seed',"$Seed",'--stand_s','4','--walk_s','8','--stop_s','6',
    '--walk_speed',$WalkSpeed.ToString('R',$taskCulture),'--lateral_speed',$LateralSpeed.ToString('R',$taskCulture),
    '--yaw_rate',$YawRate.ToString('R',$taskCulture),'--push_speed',$PushSpeed.ToString('R',$taskCulture),
    '--view',$View,'--device','cuda:0','--headless','--rendering_mode','performance','--kit_args=--/app/vulkan=false')
if ($NoVideo) {$taskArgs += '--no_video'}
if ($PreflightOnly) {
    Write-Output ('PREFLIGHT ONLY; no output or simulator: ' + $taskPython + ' ' + ($taskArgs -join ' '))
    return
}
$null = New-Item -ItemType Directory -Path $OutputDir
$taskInvocation = [ordered]@{schema='trained_balanced_gait_candidate_invocation_v1';status='running';
    started_at=(Get-Date).ToString('o');arm=$Arm;physics_mode=$PhysicsMode;seed=$Seed;args=$taskArgs;
    checkpoint=$taskCheckpoint;checkpoint_sha256=$taskCheckpointSha;source_and_inputs=$taskSnapshot;
    original_baseline_equivalence_verified=$false;promotion_performed=$false;
    notice='Trained candidate diagnostic only. Original-mode equivalence requires separate baseline comparison; no full-flow or fast-running claim.'}
$taskInvocationPath = Join-Path $OutputDir 'invocation.json'
function Save-RetentionInvocation {
    [IO.File]::WriteAllText($taskInvocationPath,($taskInvocation | ConvertTo-Json -Depth 15),[Text.UTF8Encoding]::new($false))
}
Save-RetentionInvocation
try {
    Assert-RetentionIdle
    $taskLog = Join-Path $OutputDir 'simulator.log'
    Push-Location $taskRepo
    try {
        & $taskPython @taskArgs *> $taskLog
        if ($LASTEXITCODE -ne 0) {throw "Retention evaluation failed with exit $LASTEXITCODE; see $taskLog"}
    } finally {Pop-Location}
    $taskFatal = @(rg -n -i 'CUDA error|CUDA_ERROR|out of memory|PhysX.*(error|overflow)|Pxg.*(error|overflow)|GPU.*buffer.*overflow|Traceback \(most recent call last\)|Fatal Python error|illegal memory access|\[Error\].*omni\.physx|discard.*contacts|contacts.*discard' $taskLog)
    if ($LASTEXITCODE -notin @(0,1)) {throw 'Could not audit simulator log.'}
    if ($taskFatal.Count) {throw ('Invalid simulator evidence: ' + ($taskFatal -join ' | '))}
    Assert-RetentionIntegrity
    $taskReportPath = Join-Path $OutputDir 'model_4246_stand_walk_stop.json'
    $taskReport = Get-Content -Raw -LiteralPath $taskReportPath | ConvertFrom-Json
    $taskExperiment = $taskReport.retention_experiment
    if ($taskReport.protocol_version -ne 'trained_balanced_gait_candidate_retention_v1' -or
        $taskReport.baseline_metric_protocol_version -ne 'stand_walk_stop_stance_geometry_v2' -or
        $taskReport.task -ne $taskName -or $taskReport.seed -ne $Seed -or
        $taskReport.checkpoint_sha256 -ne $taskCheckpointSha -or
        [IO.Path]::GetFullPath($taskReport.checkpoint) -ne $taskCheckpoint -or
        $taskExperiment.physics_mode -ne $PhysicsMode -or $taskExperiment.candidate_checkpoint_sha256 -ne $taskCheckpointSha -or
        $taskExperiment.template_sha256 -ne $taskTemplateSha -or
        $taskExperiment.adapter_sha256 -ne (Get-FileHash -LiteralPath $taskEntry -Algorithm SHA256).Hash -or
        $taskExperiment.no_added_reset_or_history_write -ne $true -or
        $taskExperiment.read_only_interface_steps -ne 900 -or $taskReport.global.steps -ne 900 -or
        $taskExperiment.reference_checked -ne ($PhysicsMode -eq 'recovery')) {throw 'Report identity/protocol mismatch.'}
    if ($taskReport.candidate_training.arm -cne $Arm -or
        $taskReport.candidate_training.balanced_duration_weight -ne $(if ($Arm -eq 'control') {0.0} else {-10.0}) -or
        $taskReport.candidate_training.actual_environment_steps -ne 921600 -or
        $taskReport.candidate_training.actual_control_steps -ne 7200 -or
        $taskReport.candidate_training.actual_iteration -ne 4246 -or
        $taskReport.candidate_training.checkpoint_sha256 -ne $taskCheckpointSha -or
        $taskReport.candidate_training.formal_updates_verified -ne 300 -or
        $taskReport.candidate_training.training_receipt_sha256 -ne (Get-FileHash -LiteralPath $TrainingResult).Hash -or
        $taskExperiment.parent_control_sha256 -ne '3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f' -or
        $taskReport.protocol.stand_s -ne 4 -or $taskReport.protocol.walk_s -ne 8 -or $taskReport.protocol.stop_s -ne 6 -or
        $taskReport.protocol.walk_speed -ne $WalkSpeed -or $taskReport.protocol.lateral_speed -ne $LateralSpeed -or
        $taskReport.protocol.yaw_rate -ne $YawRate -or $taskReport.protocol.push_delta_vy -ne $PushSpeed) {throw 'Command protocol mismatch.'}
    if (($NoVideo -and $null -ne $taskReport.video_view) -or (-not $NoVideo -and $taskReport.video_view -ne $View)) {throw 'Video mode mismatch.'}
    $taskCriteria = @('no_reset','no_base_contact','supported_height','level','walk_tracking','quiet_stand_stop',
        'feet_lift_in_walk','limited_slip','four_feet_at_rest','normal_stance_geometry_at_rest',
        'four_vertical_contacts_at_rest','no_current_base_contact_at_rest','geometry_and_support_at_rest')
    if ($LateralSpeed -ne 0 -or $YawRate -ne 0) {$taskCriteria += @('lateral_tracking','yaw_tracking','quiet_yaw')}
    $taskAcceptance = @($taskReport.acceptance.PSObject.Properties)
    if ((@($taskAcceptance.Name | Sort-Object) -join ',') -cne (@($taskCriteria | Sort-Object) -join ',') -or
        @($taskAcceptance | Where-Object {$_.Value -isnot [bool]}).Count) {throw 'Missing/changed/nonboolean acceptance criteria.'}
    $taskFailed = @($taskAcceptance | Where-Object {-not $_.Value} | ForEach-Object {$_.Name})
    if ($taskReport.passed -isnot [bool] -or $taskReport.passed -ne ($taskFailed.Count -eq 0)) {throw 'Inconsistent original passed flag.'}
    $taskWalk = $taskReport.settled_phase_stats.walk
    foreach ($taskMetric in @('vx_b_mean','vy_b_mean','yaw_rate_mean','contact_slip_mean','max_tilt_deg')) {
        $taskValue = $taskWalk.$taskMetric
        if ($null -eq $taskValue -or $taskValue -is [string] -or $taskValue -is [bool] -or
            [double]::IsNaN([double]$taskValue) -or [double]::IsInfinity([double]$taskValue)) {throw "Invalid metric: $taskMetric"}
    }
    $taskStraight = $LateralSpeed -eq 0 -and $YawRate -eq 0 -and $PushSpeed -eq 0
    $taskDrift = if ($taskStraight) {[Math]::Abs($taskWalk.vy_b_mean) -lt .12 -and [Math]::Abs($taskWalk.yaw_rate_mean) -lt .15} else {$null}
    $taskRetained = $taskReport.passed -and ($null -eq $taskDrift -or $taskDrift)
    if ($taskReport.retention_acceptance.original_checks -ne $taskReport.passed -or
        $taskReport.retention_acceptance.straight_drift -ne $taskDrift -or
        $taskReport.retention_passed -isnot [bool] -or $taskReport.retention_passed -ne $taskRetained) {throw 'Retention/straight-drift gate mismatch.'}
    $taskTracePath = Join-Path $OutputDir 'retention_interface.json'
    $taskGenerated = Join-Path $OutputDir 'retention_generated.py'
    if ([IO.Path]::GetFullPath($taskExperiment.trace_path) -ne $taskTracePath -or
        (Get-FileHash -LiteralPath $taskTracePath -Algorithm SHA256).Hash -ne $taskExperiment.trace_sha256 -or
        (Get-FileHash -LiteralPath $taskGenerated -Algorithm SHA256).Hash -ne $taskExperiment.generated_sha256) {throw 'Trace/generated-source provenance mismatch.'}
    $taskTrace = Get-Content -Raw -LiteralPath $taskTracePath | ConvertFrom-Json
    if ($taskTrace.schema -ne 'trained_balanced_gait_candidate_retention_v1' -or $taskTrace.physics_mode -ne $PhysicsMode -or
        @($taskTrace.steps).Count -ne 900 -or $taskTrace.interface.observation_dim -ne 48 -or
        $taskTrace.interface.step_dt -ne .02 -or $taskTrace.interface.physics_dt -ne .005 -or
        $taskTrace.config.reference_checked -ne ($PhysicsMode -eq 'recovery') -or
        $taskTrace.config.after.robot.spawn.articulation_props.enabled_self_collisions -ne ($PhysicsMode -eq 'recovery')) {throw 'Incomplete/incompatible runtime interface evidence.'}
    $taskActionClass = if ($PhysicsMode -eq 'recovery') {'ControlStepJointPositionAction'} else {'JointPositionAction'}
    if ($taskTrace.interface.action_class -ne $taskActionClass -or $taskTrace.interface.scale -ne .25 -or
        @($taskTrace.interface.native_joint_names).Count -ne 12 -or @($taskTrace.interface.default_joint_positions).Count -ne 12) {throw 'Runtime action/joint layout mismatch.'}
    for ($taskStep = 0; $taskStep -lt 900; $taskStep++) {
        $taskRow = $taskTrace.steps[$taskStep]
        if ($taskRow.step -ne $taskStep -or $taskRow.physics_mode -ne $PhysicsMode -or
            @($taskRow.observation).Count -ne 1 -or @($taskRow.observation[0]).Count -ne 48 -or
            @($taskRow.raw_action).Count -ne 1 -or @($taskRow.raw_action[0]).Count -ne 12 -or
            @($taskRow.done).Count -ne 1 -or $taskRow.done[0] -ne $false) {throw "Incomplete runtime row $taskStep"}
        foreach ($taskPair in @(@('actual_action','raw_action'),@('actual_prev_action','previous_raw_action'),@('executed_target','expected_target'))) {
            if (($taskRow.($taskPair[0]) | ConvertTo-Json -Depth 4 -Compress) -cne
                ($taskRow.($taskPair[1]) | ConvertTo-Json -Depth 4 -Compress)) {throw "Action/target/history mismatch at $taskStep"}
        }
        if ($taskStep -gt 0 -and ($taskRow.previous_raw_action | ConvertTo-Json -Depth 4 -Compress) -cne
            ($taskTrace.steps[$taskStep - 1].actual_action | ConvertTo-Json -Depth 4 -Compress)) {throw "Discontinuous action history at $taskStep"}
    }
    $taskCsv = Join-Path $OutputDir 'model_4246_stand_walk_stop.csv'
    if ([IO.Path]::GetFullPath($taskReport.artifacts.csv) -ne $taskCsv -or @(Import-Csv -LiteralPath $taskCsv).Count -ne 900) {throw 'Incomplete CSV artifact.'}
    if (-not $NoVideo) {
        $taskVideo = Join-Path $OutputDir 'model_4246_stand_walk_stop.mp4'
        if ([IO.Path]::GetFullPath($taskReport.artifacts.video) -ne $taskVideo -or
            -not (Test-Path -LiteralPath $taskVideo -PathType Leaf) -or (Get-Item -LiteralPath $taskVideo).Length -eq 0) {throw 'Missing video artifact.'}
    }
    $taskInvocation.status = 'completed_screening_only'
    $taskInvocation['report'] = $taskReportPath
    $taskInvocation['report_sha256'] = (Get-FileHash -LiteralPath $taskReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $taskInvocation['report_passed'] = $taskReport.passed
    $taskInvocation['failed_criteria'] = $taskFailed
    $taskInvocation['straight_drift_passed'] = $taskDrift
    $taskInvocation['retention_passed'] = $taskReport.retention_passed
    Write-Output "RETENTION SCREEN COMPLETE mode=$PhysicsMode seed=$Seed command=$WalkSpeed passed=$taskRetained vx=$($taskWalk.vx_b_mean). No promotion or full-flow claim."
} catch {
    $taskInvocation.status = 'error'
    $taskInvocation['error'] = $_.Exception.Message
    throw
} finally {
    $taskInvocation['finished_at'] = (Get-Date).ToString('o')
    Save-RetentionInvocation
}
