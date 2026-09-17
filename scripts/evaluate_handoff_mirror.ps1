param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$StandCheckpoint,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [Parameter(Mandatory=$true)][ValidateSet('off','initial_right')][string]$Mode,
    [ValidateSet('upright','side','upside_down')][string]$Pose = 'side',
    [ValidateSet(1,2,20)][int]$Trials = 20,
    [int]$Seed = 20260918,
    [ValidateSet('oblique','front','side')][string]$View = 'oblique',
    [switch]$Video,
    [switch]$PreflightOnly
)
# Independent frozen-weight DIAGNOSTIC. No training, promotion, ramp or retry.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 3
$taskRoot='E:\IsaacLab\go2-rl-open-source'
$taskWorkspace='E:\IsaacLab\'
$taskPython='E:\IsaacLab\env\python.exe'
$taskName='Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0'
$taskBank=Join-Path $taskRoot 'datasets\recovery_states\nominal_pd_v1_20260916'
$taskEntry=Join-Path $PSScriptRoot 'evaluate_handoff_mirror.py'
$taskMirrorMath=Join-Path $taskRoot 'src\go2_recovery\roll_mirror_math.py'
$taskVerify=Join-Path $PSScriptRoot 'verify_overnight_baselines.py'
$taskRollSha='71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c'
$taskStandSha='5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb'
$taskOldSha='4024ba713e3ed616225d64e1d59de97fd31c8369bba83ac3f4235d93552e9e2f'
$taskTransitionSha='c3a433568561fbe0893d61a570cf420304d44dcf986440e8d7efebd85ff7f10a'
$taskTransitionMathSha='b66167039b4698d66f3cc412b19bacd0b1b2f37279f7ce0856ec6c09b196aa4e'
$taskBankSha='71b882e92035b4c248e5980339c980dbb242dcb1ec711c05252c33249db48f5a'
$taskCriterion='gravity error < 0.35, height 0.30-0.55 m, speed < 0.50 m/s, angular speed < 1.00 rad/s, 4 simultaneous foot vertical forces > 5 N, no base contact, feet on correct body sides (0.06 < signed lateral < 0.30 m), knees on correct sides (>0.04 m), fore/hind feet on correct ends (>0.08 m), each joint offset <0.65 rad; all continuously held for hold_s'
$taskSelectionRule='eligible_settled_fallen_at_real_policy_start AND normalized_projected_gravity_body_y < -cos(30deg); mode off selects none; no requested pose/ID/outcome input'

function Assert-MirrorIdle {
    $taskBusy=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit.*|isaac-sim.*)\.exe$' -and
        ($_.CommandLine -match 'E:[\\/]IsaacLab' -or $_.ExecutablePath -match '^E:[\\/]IsaacLab[\\/]')
    })
    if ($taskBusy.Count) {throw "Workspace Python/Kit active: $($taskBusy.ProcessId -join ','). No overlapping simulator or CPU validator."}
}
function Get-Sha([string]$Path) {return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Assert-WorkspacePath([string]$Path) {
    $taskAbsolute=[IO.Path]::GetFullPath($Path)
    if (-not $taskAbsolute.StartsWith($taskWorkspace,[StringComparison]::OrdinalIgnoreCase)) {throw "Non-workspace path: $Path"}
    # Refuse reparse-point ancestors rather than allow an E path to redirect to C.
    $taskPart=$taskAbsolute
    while ($taskPart -and $taskPart.Length -gt 3) {
        if (Test-Path -LiteralPath $taskPart) {
            if ((Get-Item -LiteralPath $taskPart -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {throw "Reparse path refused: $taskPart"}
        }
        $taskPart=[IO.Path]::GetDirectoryName($taskPart)
    }
    return $taskAbsolute
}
function Assert-Vector($Value,[int]$Width,[string]$Label) {
    if (@($Value).Count -ne $Width) {throw "Invalid vector length: $Label"}
    foreach ($taskNumber in $Value) {
        if ($null -eq $taskNumber -or $taskNumber -is [bool] -or $taskNumber -is [string] -or
            [double]::IsNaN([double]$taskNumber) -or [double]::IsInfinity([double]$taskNumber)) {throw "Nonfinite/nonnumeric vector: $Label"}
    }
}
function Assert-ExactVector($Left,$Right,[int]$Width,[string]$Label) {
    Assert-Vector $Left $Width $Label
    Assert-Vector $Right $Width $Label
    for ($taskIndex=0;$taskIndex -lt $Width;$taskIndex++) {if ($Left[$taskIndex] -ne $Right[$taskIndex]) {throw "Exact native coordinate mismatch: $Label"}}
}
function Write-NewJson([string]$Path,$Value) {
    $taskJson=$Value | ConvertTo-Json -Depth 30
    $taskBytes=[Text.UTF8Encoding]::new($false).GetBytes($taskJson)
    $taskStream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)} finally {$taskStream.Dispose()}
}
Assert-MirrorIdle
$Checkpoint=Assert-WorkspacePath $Checkpoint
$StandCheckpoint=Assert-WorkspacePath $StandCheckpoint
$OutputDir=Assert-WorkspacePath $OutputDir
if (-not $OutputDir.StartsWith((Join-Path $taskRoot 'evaluations\'),[StringComparison]::OrdinalIgnoreCase)) {throw 'Use a fresh directory below portfolio/evaluations.'}
if (Test-Path -LiteralPath $OutputDir) {throw 'Output exists; preserve evidence and choose a new directory.'}
$taskPins=@{
    $Checkpoint=$taskRollSha; $StandCheckpoint=$taskStandSha
    (Join-Path $PSScriptRoot 'evaluate_go2_recovery.py')=$taskOldSha
    (Join-Path $PSScriptRoot 'evaluate_handoff_transition.py')=$taskTransitionSha
    (Join-Path $taskRoot 'src\go2_recovery\handoff_transition_math.py')=$taskTransitionMathSha
    (Join-Path $taskRoot 'src\go2_recovery\recovery_handoff_math.py')='df8a45a5e64ddc05a3062b73f769134fd3612aa4149f88eae51a67252699369b'
    (Join-Path $taskBank 'states.npz')=$taskBankSha
    (Join-Path $taskBank 'manifest.json')='809d90f8c2618e1ad961e2e8b5bfe33df358d20373ad85e341c26af249a342aa'
    (Join-Path $taskRoot 'configs\overnight_preservation_20260917.json')='0250a8c60db31fb21df03ffd16591d8c8e9bfe61e1c0d85a76e91c2929648e55'
}
foreach ($taskPin in $taskPins.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $taskPin.Key -PathType Leaf) -or (Get-Sha $taskPin.Key) -ne $taskPin.Value) {throw "Frozen input mismatch: $($taskPin.Key)"}
}
if ($Checkpoint -eq $StandCheckpoint) {throw 'Distinct frozen actors are required.'}
$taskPaths=@($taskPins.Keys)+@($PSCommandPath,$taskEntry,$taskMirrorMath,$taskVerify,
    (Join-Path $PSScriptRoot 'compare_handoff_mirror_baseline.py'),
    (Join-Path $PSScriptRoot 'compare_handoff_transition_baseline.py'),
    'E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd',
    'E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd')
$taskInstalled='E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$taskPaths+=@(rg --files -g '*.py' $taskInstalled)
if ($LASTEXITCODE -ne 0) {throw 'Cannot inventory runtime Go2 dependencies.'}
$taskSnapshot=@($taskPaths | Sort-Object -Unique | ForEach-Object {
    $taskPath=Assert-WorkspacePath $_
    if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) {throw "Missing provenance input: $taskPath"}
    @{path=$taskPath;sha256=(Get-Sha $taskPath)}
})
function Assert-Snapshot {
    foreach ($taskItem in $taskSnapshot) {if ((Get-Sha $taskItem.path) -ne $taskItem.sha256) {throw "Source/input drift: $($taskItem.path)"}}
}
$taskAngle=if ($Pose -eq 'upright') {'0'} else {'30'}
$taskArgs=@('-B',$taskEntry,'--task',$taskName,'--checkpoint',$Checkpoint,'--stand_checkpoint',$StandCheckpoint,
    '--output_dir',$OutputDir,'--mirror_mode',$Mode,'--trials',"$Trials",'--seed',"$Seed",'--poses',$Pose,
    '--angle_deg',$taskAngle,'--settle_s','1','--horizon_s','8','--hold_s','3','--min_contacts','4',
    '--device','cuda:0','--headless','--kit_args=--/app/vulkan=false')
if ($Pose -ne 'upright') {$taskArgs+=@('--state_bank_path',$taskBank,'--state_bank_split','heldout')}
if ($Video) {$taskArgs+=@('--video_pose','all','--view',$View)}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP='E:\IsaacLab\tmp';$env:TMP='E:\IsaacLab\tmp'
& $taskPython -B $taskVerify
if ($LASTEXITCODE -ne 0) {throw 'Frozen16model preservation failed before experiment.'}
if ($PreflightOnly) {Write-Output ('PREFLIGHT ONLY: '+$taskPython+' '+($taskArgs -join ' '));return}
Assert-MirrorIdle
Assert-Snapshot
$null=New-Item -ItemType Directory -Path $OutputDir
Write-NewJson (Join-Path $OutputDir 'invocation.json') @{
    schema='handoff_mirror_invocation_v1';started_at=(Get-Date).ToString('o');mode=$Mode;pose=$Pose;trials=$Trials;seed=$Seed;
    args=$taskArgs;source_and_inputs=$taskSnapshot;acceptance_eligible=$false;training_collection_eligible=$false;
    notice='FROZEN-WEIGHT MIRROR DIAGNOSTIC ONLY; no training, model promotion, ramp, retry or startup selection.'
}
$env:OMNI_KIT_ACCEPT_EULA='YES';$env:OMNI_USER_HOME='E:\IsaacLab\userdata';$env:OV_USER_HOME='E:\IsaacLab\userdata'
$env:PIP_CACHE_DIR='E:\IsaacLab\cache\pip';$env:PYTHONIOENCODING='utf-8'
$env:ISAACLAB_GO2_USD='E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd'
$env:ISAACLAB_GROUND_USD='E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd'
$env:CONDA_PREFIX='E:\IsaacLab\env';$env:Path='E:\IsaacLab\env;E:\IsaacLab\env\Scripts;'+$env:Path
$taskPreviousBank=[Environment]::GetEnvironmentVariable('ISAACLAB_RECOVERY_BANK_COLLECTION','Process')
$taskReceipt=@{schema='handoff_mirror_wrapper_validation_v1';completed_diagnostic=$false;acceptance_eligible=$false;
    training_collection_eligible=$false;promotion_performed=$false;mode=$Mode;pose=$Pose;trials=$Trials;seed=$Seed;
    hard_equivalence_proven=$false;notice='Schema/provenance/trace checks only. OFF requires separate exact matcher; ON is diagnostic, not acceptance.'}
$taskSimLog=Join-Path $OutputDir 'simulator.log'
try {
    $env:ISAACLAB_RECOVERY_BANK_COLLECTION='1'
    Assert-MirrorIdle
    & $taskPython @taskArgs *> $taskSimLog
    if ($LASTEXITCODE -ne 0) {throw "Mirror evaluator exited with $LASTEXITCODE"}
    $taskFatal=@(rg -n -i 'CUDA error|CUDA_ERROR|out of memory|PhysX.*(error|overflow)|Pxg.*(error|overflow)|GPU.*buffer.*overflow|Traceback \(most recent call last\)|Fatal Python error|illegal memory access|\[Error\].*omni\.physx|discard.*contacts|contacts.*discard' $taskSimLog)
    if ($LASTEXITCODE -notin @(0,1)) {throw 'Could not audit simulator log.'}
    if ($taskFatal.Count) {throw ('Simulation failure evidence: '+($taskFatal -join ' | '))}
    if (Test-Path -LiteralPath (Join-Path $OutputDir 'transition_error.json')) {throw 'Evaluator recorded an exception before shutdown.'}
    Assert-Snapshot
    $taskReportPath=Join-Path $OutputDir ((Split-Path $Checkpoint -Leaf)+'_recovery_metrics.json')
    $r=Get-Content -Raw -LiteralPath $taskReportPath | ConvertFrom-Json
    if ($r.protocol_version -cne 'handoff_mirror_experimental_v1' -or
        $r.controller_type -cne 'experimental_roll_only_initial_side_mirror_hard_handoff' -or
        $r.baseline_transition_protocol_version -cne 'handoff_transition_experimental_v1' -or
        $r.task -cne $taskName -or $r.seed -ne $Seed -or $r.criterion -cne $taskCriterion -or
        $r.policy_action_mode -cne 'deterministic_mean' -or $r.self_collisions_enabled -ne $true -or
        $r.settle_requested_s -ne 1 -or $r.settle_actual_s -ne 1 -or $r.settle_control_steps -ne 50) {throw 'Mirror report task/physics/protocol mismatch.'}
    foreach ($taskFlag in @('acceptance_eligible','single_policy_acceptance_eligible','training_collection_eligible')) {
        if ($r.$taskFlag -isnot [bool] -or $r.$taskFlag) {throw "Invalid nonacceptance flag: $taskFlag"}
    }
    if ($r.checkpoint -cne $Checkpoint -or $r.checkpoint_sha256 -cne $taskRollSha -or
        $r.handoff_controller.roll_checkpoint -cne $Checkpoint -or $r.handoff_controller.roll_sha256 -cne $taskRollSha -or
        $r.handoff_controller.stand_checkpoint -cne $StandCheckpoint -or $r.handoff_controller.stand_sha256 -cne $taskStandSha -or
        $r.handoff_controller.gate -cne 'tilt<30deg and body angular speed<1rad/s continuously for0.2s; one-way latched' -or
        $r.handoff_controller.state_or_action_history_reset_at_switch -ne $false -or
        $r.handoff_controller.manual_standing_pd_at_switch -ne $false -or $r.handoff_controller.physics_changed_at_switch -ne $false) {throw 'Frozen actor identity or one-way gate changed.'}
    if ($r.action_representation.reference -cne 'nominal' -or $r.action_representation.scale -ne .25 -or
        $r.action_representation.sample_period_s -ne .02 -or $r.action_representation.held_over_physics_substeps -ne 4) {throw 'Action representation changed.'}
    $t=$r.transition_experiment;$m=$r.mirror_experiment
    if ($t.mode -cne 'hard' -or $t.ramp_seconds -ne 0 -or $t.ramp_steps -ne 0 -or
        $t.baseline_evaluator_sha256 -cne $taskOldSha -or $t.math_source_sha256 -cne $taskTransitionMathSha -or
        $t.evaluator_source_sha256 -cne (Get-Sha $taskEntry) -or $m.evaluator_source_sha256 -cne (Get-Sha $taskEntry) -or
        $m.math_source_sha256 -cne (Get-Sha $taskMirrorMath) -or $m.baseline_transition_evaluator_sha256 -cne $taskTransitionSha -or
        $m.baseline_transition_math_sha256 -cne $taskTransitionMathSha -or $m.mode -cne $Mode -or
        $m.selection_rule -cne $taskSelectionRule -or $m.normalized_gravity_y_threshold -ne (-[Math]::Cos([Math]::PI/6))) {throw 'Mirror source or selection contract mismatch.'}
    foreach ($e in @($t,$m)) {
        if ($e.step_dt -ne .02 -or $e.horizon_s -ne 8 -or $e.hold_s -ne 3 -or $e.min_contacts -ne 4 -or $e.policy_control_steps -ne 550) {throw 'Timing/strict-contact protocol changed.'}
    }
    foreach ($flag in @('fixed_mask_per_episode','roll_only','standing_uses_real_observation','gate_uses_real_state','real_policy48_asserted_against_runtime_every_step')) {if ($m.$flag -isnot [bool] -or -not $m.$flag) {throw "Missing required mirror guarantee: $flag"}}
    foreach ($flag in @('physics_or_history_mutated','ramp_enabled','retry_enabled','startup_selection_changed','physical_asset_symmetry_proven','paper_training_reproduction')) {if ($m.$flag -isnot [bool] -or $m.$flag) {throw "Forbidden mirror intervention/claim: $flag"}}
    if (@($r.results.PSObject.Properties).Count -ne 1 -or $r.results.$Pose.trials -ne $Trials) {throw 'Incomplete or changed pose denominator.'}
    $p=$r.results.$Pose
    foreach ($countName in @('successes','final_valid_stands','final_geometry_passes','settled_fallen_trials')) {
        $count=$p.$countName
        if (($count -isnot [int] -and $count -isnot [long]) -or $count -lt 0 -or $count -gt $Trials) {throw "Invalid count: $countName"}
    }
    if ($p.final_valid_stands -gt $p.successes -or $p.final_valid_stands -gt $p.final_geometry_passes -or $p.horizon_s -ne 8 -or $p.stable_hold_s -ne 3) {throw 'Invalid final result/hold denominator.'}
    foreach ($name in @('final_diagnostics','policy_start_state','release_state_before_settling')) {if (@($p.$name).Count -ne $Trials) {throw "Missing per-trial state: $name"}}
    if ($Pose -ne 'upright' -and ($r.state_bank.split -cne 'heldout' -or $r.state_bank.sha256 -cne $taskBankSha -or @($r.state_bank.selected_state_ids).Count -ne $Trials)) {throw 'Frozen heldout bank mismatch.'}
    $taskTraceName=(Split-Path $Checkpoint -Leaf)+'_handoff_mirror_trace.json'
    if ($m.mirror_trace_file -cne $taskTraceName -or $t.transition_trace_file -cne $taskTraceName) {throw 'Trace must be local to the new case directory.'}
    $taskTracePath=Join-Path $OutputDir $taskTraceName
    if ((Get-Sha $taskTracePath) -cne $m.mirror_trace_sha256 -or $t.transition_trace_sha256 -cne $m.mirror_trace_sha256) {throw 'Mirror trace SHA mismatch.'}
    $trace=Get-Content -Raw -LiteralPath $taskTracePath | ConvertFrom-Json
    if ($trace.schema -cne 'handoff_mirror_neighborhood_v1' -or $trace.checkpoint_sha256 -cne $taskRollSha -or
        $trace.stand_checkpoint_sha256 -cne $taskStandSha -or $trace.mirror_experiment.mode -cne $Mode -or
        @($trace.poses.PSObject.Properties).Count -ne 1 -or @($trace.poses.$Pose).Count -ne $Trials) {throw 'Incomplete all-trial mirror trace.'}
    if (@($p.mirror_diagnostic.selection_records).Count -ne $Trials -or $p.mirror_diagnostic.total_trials -ne $Trials -or
        $p.mirror_diagnostic.all_trials_in_success_denominator -ne $true) {throw 'Mirror mask denominator changed.'}
    $taskSelected=0
    for ($i=0;$i -lt $Trials;$i++) {
        $s=$p.mirror_diagnostic.selection_records[$i];$start=$p.policy_start_state[$i];$trial=$trace.poses.$Pose[$i]
        if ($s.trial -ne $i -or $start.trial -ne $i -or $p.final_diagnostics[$i].trial -ne $i -or $trial.trial -ne $i -or
            $s.selected -isnot [bool] -or $s.mask_latched_for_episode -ne $true -or @($trial.rows).Count -eq 0) {throw 'Missing/reordered trial or fixed mask.'}
        Assert-ExactVector $s.real_policy_start_projected_gravity_b $start.projected_gravity_b 3 'real startup gravity'
        if ($s.eligible_settled_fallen_at_policy_start -isnot [bool] -or $s.eligible_settled_fallen_at_policy_start -ne $start.eligible_settled_fallen_recovery -or
            $s.strict_gravity_y_threshold -ne $m.normalized_gravity_y_threshold) {throw 'Mask uses other than actual eligible startup state.'}
        $gy=$s.normalized_policy_start_gravity_y
        Assert-Vector @($gy) 1 'normalized initial gravity'
        $g=$start.projected_gravity_b;$norm=[Math]::Sqrt($g[0]*$g[0]+$g[1]*$g[1]+$g[2]*$g[2])
        if ($norm -le 0 -or [Math]::Abs($gy-$g[1]/$norm) -gt 1e-6) {throw 'Recorded normalized initial gravity contradicts real state.'}
        $expected=($Mode -eq 'initial_right' -and $start.eligible_settled_fallen_recovery -and $gy -lt $m.normalized_gravity_y_threshold)
        if ($s.selected -ne $expected) {throw 'Fixed actual-state right-cone mask mismatch; no pose-label shortcut allowed.'}
        if ($s.selected) {$taskSelected++}
        if (($trial.mirror_selection | ConvertTo-Json -Compress -Depth 10) -cne ($s | ConvertTo-Json -Compress -Depth 10)) {throw 'Trace/report mask records differ.'}
        $lastStep=-1
        foreach ($row in $trial.rows) {
            if ($row.trial -ne $i -or $row.step -le $lastStep -or $row.roll_mirror_selected -ne $s.selected -or
                $row.issued_action_target_history_assertions_passed -ne $true) {throw 'Invalid, reordered or mask-changing trace row.'}
            $lastStep=$row.step
            foreach ($name in @('policy_observation','real_policy_observation','roll_policy_input_observation')) {Assert-Vector $row.$name 48 $name}
            foreach ($name in @('roll_actor_raw_action','stand_actor_raw_action','roll_actor_output_model_raw_action','roll_actor_output_physical_raw_action','issued_raw_action','actual_raw_action','previous_raw_action','previous_previous_raw_action','actual_previous_raw_action','actual_executed_joint_target_rad','expected_executed_joint_target_rad')) {Assert-Vector $row.$name 12 $name}
            Assert-ExactVector $row.real_policy_observation $row.policy_observation 48 'real legacy observation'
            Assert-ExactVector $row.roll_actor_output_physical_raw_action $row.roll_actor_raw_action 12 'native roll candidate'
            Assert-ExactVector $row.actual_raw_action $row.issued_raw_action 12 'actually issued action'
            Assert-ExactVector $row.actual_previous_raw_action $row.previous_raw_action 12 'natural history advance'
            if (-not $s.selected) {
                Assert-ExactVector $row.roll_policy_input_observation $row.policy_observation 48 'unselected real roll input'
                Assert-ExactVector $row.roll_actor_output_model_raw_action $row.roll_actor_raw_action 12 'unselected native output'
            }
        }
    }
    if ($p.mirror_diagnostic.selected_trials -ne $taskSelected) {throw 'Selected count contradicts all-trial mask records.'}
    Assert-MirrorIdle
    & $taskPython -B $taskVerify
    if ($LASTEXITCODE -ne 0) {throw 'Frozen16model preservation failed after experiment.'}
    Assert-Snapshot
    $taskReceipt.completed_diagnostic=$true
    $taskReceipt.final_valid_stands=$p.final_valid_stands
    $taskReceipt.selected_trials=$taskSelected
    $taskReceipt.report_sha256=Get-Sha $taskReportPath
    $taskReceipt.trace_sha256=Get-Sha $taskTracePath
    $taskReceipt.source_and_inputs=$taskSnapshot
} catch {
    $taskReceipt.error=$_.Exception.Message
    throw
} finally {
    if ($null -eq $taskPreviousBank) {Remove-Item Env:ISAACLAB_RECOVERY_BANK_COLLECTION -ErrorAction SilentlyContinue}
    else {$env:ISAACLAB_RECOVERY_BANK_COLLECTION=$taskPreviousBank}
    $taskReceipt.finished_at=(Get-Date).ToString('o')
    if (Test-Path -LiteralPath $taskSimLog -PathType Leaf) {$taskReceipt.simulator_log_sha256=Get-Sha $taskSimLog}
    Write-NewJson (Join-Path $OutputDir 'wrapper_validation.json') $taskReceipt
}
Write-Output "MIRROR DIAGNOSTIC COMPLETE mode=$Mode pose=$Pose final=$($taskReceipt.final_valid_stands)/$Trials; selected=$($taskReceipt.selected_trials). OFF still requires exact matcher; ON is not acceptance."
