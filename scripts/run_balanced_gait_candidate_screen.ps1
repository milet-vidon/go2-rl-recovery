param(
    [Parameter(Mandatory=$true)][ValidateSet('control','balanced')][string]$Arm,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$TrainingResult
)
# One finite21-case final4246 screen. Every case is newly executed, including .8.
# No training, reuse, threshold change, promotion or resume/overwrite behavior.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$Arm=$Arm.ToLowerInvariant()
$taskRoot='E:\IsaacLab\go2-rl-open-source'
$taskRunRoot='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_flat\'
$Checkpoint=(Resolve-Path -LiteralPath $Checkpoint).Path
$TrainingResult=(Resolve-Path -LiteralPath $TrainingResult).Path
if(-not $Checkpoint.StartsWith($taskRunRoot,[StringComparison]::OrdinalIgnoreCase) -or
   (Split-Path $Checkpoint -Leaf) -cne 'model_4246.pt' -or
   (Split-Path $TrainingResult -Leaf) -cne 'common_physics_training_result.json' -or
   (Split-Path $Checkpoint) -ne (Split-Path $TrainingResult)){throw 'Require same-run isolated final4246 and actual formal training receipt.'}
$taskOutput=Join-Path $taskRoot "evaluations\20260917-gait-$Arm-4246-first"
if(Test-Path -LiteralPath $taskOutput){throw 'Preserve existing screen output; this batch never resumes or reuses cases.'}
$taskSummaryPath=Join-Path $taskOutput 'summary.json'
$taskEvaluator=Join-Path $PSScriptRoot 'evaluate_balanced_gait_candidate.ps1'
$taskEntry=Join-Path $PSScriptRoot 'evaluate_balanced_gait_candidate.py'
$taskWeight=if($Arm -eq 'control'){0.0}else{-10.0}
$taskReceipt=Get-Content -Raw -LiteralPath $TrainingResult|ConvertFrom-Json
if($taskReceipt.protocol -cne 'balanced_gait_common_physics_training_v1' -or
   $taskReceipt.completion_verified -isnot [bool] -or -not $taskReceipt.completion_verified -or
   $taskReceipt.arm -cne $Arm -or $taskReceipt.balanced_duration_weight -ne $taskWeight -or
   $taskReceipt.num_envs -ne 128 -or $taskReceipt.updates -ne 300 -or
   $taskReceipt.actual_environment_steps -ne 921600 -or $taskReceipt.actual_control_steps -ne 7200 -or
   [IO.Path]::GetFullPath($taskReceipt.checkpoint) -ne $Checkpoint -or
   $taskReceipt.quality_accepted -ne $false -or $taskReceipt.promotion_performed -ne $false){throw 'Wrong/incomplete arm-specific formal300 receipt. Full actual checkpoint verification is also mandatory in every case wrapper.'}
$taskSourcePaths=@($PSCommandPath,$taskEvaluator,$taskEntry,
    (Join-Path $PSScriptRoot 'test_balanced_gait_candidate.py'),
    (Join-Path $PSScriptRoot 'evaluate_common_physics_candidate.py'),
    (Join-Path $PSScriptRoot 'evaluate_locomotion_recovery_physics.py'),
    (Join-Path $PSScriptRoot 'evaluate_go2_stand_walk_stop.py'),
    (Join-Path $PSScriptRoot 'train_balanced_gait_adaptation.py'),
    (Join-Path $PSScriptRoot 'train_common_physics_adaptation.py'),
    (Join-Path $PSScriptRoot 'balanced_gait_reward.py'),
    (Join-Path $PSScriptRoot 'balanced_gait_exposure.py'),$Checkpoint,$TrainingResult)
$taskSources=@($taskSourcePaths|Sort-Object -Unique|ForEach-Object{
    @{path=[IO.Path]::GetFullPath($_);sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$taskCheckpointSha=(Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
$taskReceiptSha=(Get-FileHash -LiteralPath $TrainingResult -Algorithm SHA256).Hash.ToLowerInvariant()
$taskCases=@(
    @{name='normal';speed=.5;yaw=0.;push=0.},
    @{name='retained08';speed=.8;yaw=0.;push=0.},
    @{name='left';speed=.5;yaw=.5;push=0.},
    @{name='right';speed=.5;yaw=-.5;push=0.},
    @{name='push05';speed=.5;yaw=0.;push=.5},
    @{name='target10';speed=1.;yaw=0.;push=0.}
)
$taskSummary=[ordered]@{protocol='balanced_gait_candidate21_screen_v1';status='running';arm=$Arm;
    checkpoint=$Checkpoint;checkpoint_sha256=$taskCheckpointSha;training_receipt=$TrainingResult;
    training_receipt_sha256=$taskReceiptSha;source_snapshot=$taskSources;started_at=(Get-Date).ToString('o');
    rows=@();promotion_performed=$false;training_performed=$false;reused_cases=0;
    scope='6 newly executed common-physics cases at seed20260909, plus15 newly executed original-physics old-function cases over3seeds. Same21-case protocol as4046.1.0 is diagnostic, not fast-running certification. No integrated flow.'}
function Save-GaitScreen {
    [IO.File]::WriteAllText($taskSummaryPath,($taskSummary|ConvertTo-Json -Depth 18),[Text.UTF8Encoding]::new($false))
}
function Assert-GaitScreenSources {
    foreach($taskItem in $taskSources){
        if((Get-FileHash -LiteralPath $taskItem.path -Algorithm SHA256).Hash -ne $taskItem.sha256){throw "Screen source/input drift: $($taskItem.path)"}
    }
}
function Assert-InvocationArg($Argv,[string]$Key,[string]$Value) {
    $taskIndices=@(for($taskIndex=0;$taskIndex -lt $Argv.Count;$taskIndex++){if($Argv[$taskIndex] -ceq $Key){$taskIndex}})
    if($taskIndices.Count -ne 1 -or $taskIndices[0]+1 -ge $Argv.Count -or $Argv[$taskIndices[0]+1] -cne $Value){throw "Wrapper invocation arg mismatch: $Key"}
}
$null=New-Item -ItemType Directory -Path $taskOutput
Save-GaitScreen
try {
    foreach($taskPhysics in @('recovery','original')) {
        $taskSeeds=if($taskPhysics -eq 'recovery'){@(20260909)}else{@(20260909,20260910,20260911)}
        foreach($taskCase in $taskCases) {
            if($taskPhysics -eq 'original' -and $taskCase.name -eq 'target10'){continue}
            foreach($taskSeed in $taskSeeds) {
                Assert-GaitScreenSources
                $taskName="$taskPhysics-$($taskCase.name)-seed$taskSeed"
                $taskSummary['current_case']=$taskName
                Save-GaitScreen
                $taskDirectory=Join-Path $taskOutput $taskName
                if(Test-Path -LiteralPath $taskDirectory){throw "Case already exists; no reuse: $taskName"}
                $taskLog=Join-Path $taskOutput "$taskName.wrapper.log"
                & $taskEvaluator -Arm $Arm -Checkpoint $Checkpoint -TrainingResult $TrainingResult -PhysicsMode $taskPhysics `
                    -OutputDir $taskDirectory -WalkSpeed $taskCase.speed -LateralSpeed 0 -YawRate $taskCase.yaw `
                    -PushSpeed $taskCase.push -Seed $taskSeed -NoVideo *> $taskLog
                if($LASTEXITCODE -ne 0){throw "Case failed operationally: $taskName; see $taskLog"}
                Assert-GaitScreenSources
                $taskInvocationPath=Join-Path $taskDirectory 'invocation.json'
                $taskInvocation=Get-Content -Raw -LiteralPath $taskInvocationPath|ConvertFrom-Json
                if($taskInvocation.schema -cne 'trained_balanced_gait_candidate_invocation_v1' -or
                   $taskInvocation.status -cne 'completed_screening_only' -or $taskInvocation.arm -cne $Arm -or
                   $taskInvocation.physics_mode -cne $taskPhysics -or $taskInvocation.seed -ne $taskSeed -or
                   [IO.Path]::GetFullPath($taskInvocation.checkpoint) -ne $Checkpoint -or
                   $taskInvocation.checkpoint_sha256 -ne $taskCheckpointSha -or
                   $taskInvocation.promotion_performed -ne $false){throw 'No matching completed actual wrapper invocation.'}
                Assert-InvocationArg $taskInvocation.args '--arm' $Arm
                Assert-InvocationArg $taskInvocation.args '--candidate_training_result' $TrainingResult
                Assert-InvocationArg $taskInvocation.args '--checkpoint' $Checkpoint
                Assert-InvocationArg $taskInvocation.args '--physics_mode' $taskPhysics
                Assert-InvocationArg $taskInvocation.args '--output_dir' $taskDirectory
                Assert-InvocationArg $taskInvocation.args '--seed' ([string]$taskSeed)
                if(@($taskInvocation.args|Where-Object {$_ -ceq '--no_video'}).Count -ne 1){throw 'Screen must be a fresh nonvideo evaluation.'}
                foreach($taskInput in $taskInvocation.source_and_inputs){
                    if((Get-FileHash -LiteralPath $taskInput.path -Algorithm SHA256).Hash -ne $taskInput.sha256){throw 'Actual case source/input changed.'}
                }
                $taskReportPath=Join-Path $taskDirectory 'model_4246_stand_walk_stop.json'
                if([IO.Path]::GetFullPath($taskInvocation.report) -ne $taskReportPath -or
                   (Get-FileHash -LiteralPath $taskReportPath -Algorithm SHA256).Hash -ne $taskInvocation.report_sha256){throw 'Report path/hash mismatch.'}
                $taskReport=Get-Content -Raw -LiteralPath $taskReportPath|ConvertFrom-Json
                $taskExperiment=$taskReport.retention_experiment
                $taskProof=$taskReport.candidate_training
                if($taskReport.protocol_version -cne 'trained_balanced_gait_candidate_retention_v1' -or
                   $taskReport.baseline_metric_protocol_version -cne 'stand_walk_stop_stance_geometry_v2' -or
                   $taskExperiment.physics_mode -cne $taskPhysics -or $taskReport.seed -ne $taskSeed -or
                   $taskReport.checkpoint_sha256 -ne $taskCheckpointSha -or
                   $taskReport.protocol.stand_s -ne 4 -or $taskReport.protocol.walk_s -ne 8 -or $taskReport.protocol.stop_s -ne 6 -or
                   $taskReport.protocol.walk_speed -ne $taskCase.speed -or $taskReport.protocol.lateral_speed -ne 0 -or
                   $taskReport.protocol.yaw_rate -ne $taskCase.yaw -or $taskReport.protocol.push_delta_vy -ne $taskCase.push -or
                   $taskReport.global.steps -ne 900 -or $taskExperiment.read_only_interface_steps -ne 900 -or
                   $taskExperiment.no_added_reset_or_history_write -ne $true -or
                   $taskExperiment.reference_checked -ne ($taskPhysics -eq 'recovery')){throw 'Case identity/unchanged physical protocol mismatch.'}
                if($taskProof.arm -cne $Arm -or $taskProof.balanced_duration_weight -ne $taskWeight -or
                   $taskProof.formal_updates_verified -ne 300 -or $taskProof.actual_environment_steps -ne 921600 -or
                   $taskProof.actual_control_steps -ne 7200 -or $taskProof.actual_iteration -ne 4246 -or
                   @($taskProof.actual_adam_steps).Count -ne 17 -or
                   @($taskProof.actual_adam_steps|Where-Object {$_ -ne 85120}).Count -ne 0 -or
                   $taskProof.model_and_optimizer_all_finite -ne $true -or $taskProof.quality_accepted -ne $false -or
                   $taskProof.promotion_performed -ne $false -or $taskProof.checkpoint_sha256 -ne $taskCheckpointSha -or
                   $taskProof.training_receipt_sha256 -ne $taskReceiptSha -or
                   $taskProof.parent_control_sha256 -ne '3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f'){throw 'Actual formal300 full-state proof differs.'}
                $taskCriteria=@('no_reset','no_base_contact','supported_height','level','walk_tracking','quiet_stand_stop',
                    'feet_lift_in_walk','limited_slip','four_feet_at_rest','normal_stance_geometry_at_rest',
                    'four_vertical_contacts_at_rest','no_current_base_contact_at_rest','geometry_and_support_at_rest')
                if($taskCase.yaw -ne 0){$taskCriteria+=@('lateral_tracking','yaw_tracking','quiet_yaw')}
                $taskAcceptance=@($taskReport.acceptance.PSObject.Properties)
                if((@($taskAcceptance.Name|Sort-Object)-join ',') -cne (@($taskCriteria|Sort-Object)-join ',') -or
                   @($taskAcceptance|Where-Object {$_.Value -isnot [bool]}).Count){throw 'Acceptance fields/types changed.'}
                $taskFailed=@($taskAcceptance|Where-Object {-not $_.Value}|ForEach-Object {$_.Name})
                if($taskReport.passed -isnot [bool] -or $taskReport.passed -ne ($taskFailed.Count -eq 0)){throw 'Inconsistent original passed flag.'}
                $taskWalk=$taskReport.settled_phase_stats.walk
                foreach($taskMetric in @('vx_b_mean','vy_b_mean','yaw_rate_mean','contact_slip_mean')){
                    $taskValue=$taskWalk.$taskMetric
                    if($null -eq $taskValue -or $taskValue -is [string] -or $taskValue -is [bool] -or
                       [double]::IsNaN([double]$taskValue) -or [double]::IsInfinity([double]$taskValue)){throw "Invalid actual metric: $taskMetric"}
                }
                $taskStraight=$taskCase.yaw -eq 0 -and $taskCase.push -eq 0
                $taskDrift=if($taskStraight){[Math]::Abs($taskWalk.vy_b_mean) -lt .12 -and [Math]::Abs($taskWalk.yaw_rate_mean) -lt .15}else{$null}
                $taskPassed=$taskReport.passed -and ($null -eq $taskDrift -or $taskDrift)
                if($taskReport.retention_passed -isnot [bool] -or $taskReport.retention_passed -ne $taskPassed -or
                   $taskReport.retention_acceptance.original_checks -ne $taskReport.passed -or
                   $taskReport.retention_acceptance.straight_drift -ne $taskDrift -or
                   $taskInvocation.retention_passed -ne $taskPassed){throw 'Retention/strict straight-drift gate mismatch.'}
                if($taskDrift -eq $false){$taskFailed+='straight_drift'}
                $taskRest=$taskReport.rest_stance_criterion
                if($taskRest.geometry_required_fraction -ne 1 -or $taskRest.vertical_support_required_fraction_exclusive -ne .95 -or
                   $taskRest.vertical_force_threshold_n_exclusive -ne 5 -or $taskRest.base_contact_force_threshold_n_exclusive -ne 1 -or
                   $taskRest.settling_exclusion_s -ne 1){throw 'Current-force/rest criteria changed.'}
                $taskTracePath=Join-Path $taskDirectory 'retention_interface.json'
                $taskGeneratedPath=Join-Path $taskDirectory 'retention_generated.py'
                $taskCsv=Join-Path $taskDirectory 'model_4246_stand_walk_stop.csv'
                if([IO.Path]::GetFullPath($taskExperiment.trace_path) -ne $taskTracePath -or
                   (Get-FileHash -LiteralPath $taskTracePath -Algorithm SHA256).Hash -ne $taskExperiment.trace_sha256 -or
                   (Get-FileHash -LiteralPath $taskGeneratedPath -Algorithm SHA256).Hash -ne $taskExperiment.generated_sha256 -or
                   [IO.Path]::GetFullPath($taskReport.artifacts.csv) -ne $taskCsv -or
                   @(Import-Csv -LiteralPath $taskCsv).Count -ne 900 -or
                   $null -ne $taskReport.artifacts.video -or $null -ne $taskReport.video_view){throw 'Actual trace/generated/CSV evidence missing or changed.'}
                $taskTrace=Get-Content -Raw -LiteralPath $taskTracePath|ConvertFrom-Json
                if($taskTrace.schema -cne 'trained_balanced_gait_candidate_retention_v1' -or
                   $taskTrace.physics_mode -cne $taskPhysics -or @($taskTrace.steps).Count -ne 900 -or
                   $taskTrace.interface.observation_dim -ne 48 -or $taskTrace.interface.step_dt -ne .02 -or
                   $taskTrace.interface.physics_dt -ne .005){throw 'Incomplete actual900-step interface trace.'}
                $taskSummary.rows+=@{arm=$Arm;physics=$taskPhysics;case=$taskCase.name;seed=$taskSeed;reused_existing=$false;
                    passed=$taskPassed;original_checks=$taskReport.passed;straight_drift=$taskDrift;failed_criteria=$taskFailed;
                    vx=$taskWalk.vx_b_mean;vy=$taskWalk.vy_b_mean;yaw=$taskWalk.yaw_rate_mean;slip=$taskWalk.contact_slip_mean;
                    invocation=$taskInvocationPath;invocation_sha256=(Get-FileHash -LiteralPath $taskInvocationPath -Algorithm SHA256).Hash.ToLowerInvariant();
                    report=$taskReportPath;report_sha256=(Get-FileHash -LiteralPath $taskReportPath -Algorithm SHA256).Hash.ToLowerInvariant();
                    trace_sha256=$taskExperiment.trace_sha256;generated_sha256=$taskExperiment.generated_sha256;
                    csv_sha256=(Get-FileHash -LiteralPath $taskCsv -Algorithm SHA256).Hash.ToLowerInvariant()}
                Save-GaitScreen
                Write-Output "SCREEN arm=$Arm $taskName passed=$taskPassed vx=$($taskWalk.vx_b_mean) vy=$($taskWalk.vy_b_mean)"
            }
        }
    }
    if($taskSummary.rows.Count -ne 21){throw 'Incomplete21-case screen.'}
    Assert-GaitScreenSources
    $taskSummary['common_physics_passed']=@($taskSummary.rows|Where-Object {$_.physics -eq 'recovery' -and $_.passed}).Count
    $taskSummary['original_physics_retained']=@($taskSummary.rows|Where-Object {$_.physics -eq 'original' -and $_.passed}).Count
    $taskSummary['all21_passed']=@($taskSummary.rows|Where-Object {-not $_.passed}).Count -eq 0
    $taskSummary['current_case']=$null
    $taskSummary.status='completed_screening_only'
} catch {
    $taskSummary.status='error'
    $taskSummary['error']=$_.Exception.Message
    throw
} finally {
    $taskSummary['finished_at']=(Get-Date).ToString('o')
    Save-GaitScreen
}
