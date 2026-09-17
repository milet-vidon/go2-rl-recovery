param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$TrainingResult
)
# Finite21-case screen:6 common-physics deterministic command cases (one reused),
# plus15 old-function original-physics seed cases. Never train/promote/overwrite.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$taskRoot='E:\IsaacLab\go2-rl-open-source'
$taskOutput=Join-Path $taskRoot 'evaluations\20260917-commonphysics4046-first'
$taskSummaryPath=Join-Path $taskOutput 'summary.json'
if(Test-Path -LiteralPath $taskSummaryPath){throw 'Preserve existing summary; inspect it, do not repeat this finite batch.'}
$taskEvaluator=Join-Path $PSScriptRoot 'evaluate_common_physics_candidate.ps1'
$taskSourcePaths=@($PSCommandPath,$taskEvaluator,(Join-Path $PSScriptRoot 'evaluate_common_physics_candidate.py'),
    (Join-Path $PSScriptRoot 'test_common_physics_candidate.py'),
    (Join-Path $PSScriptRoot 'evaluate_locomotion_recovery_physics.py'),
    (Join-Path $PSScriptRoot 'train_common_physics_adaptation.py'),$Checkpoint,$TrainingResult)
$taskSources=@($taskSourcePaths|ForEach-Object{@{path=[IO.Path]::GetFullPath($_);sha256=(Get-FileHash -LiteralPath $_).Hash.ToLowerInvariant()}})
$taskCases=@(
    @{name='normal';speed=.5;yaw=0.;push=0.},
    @{name='retained08';speed=.8;yaw=0.;push=0.},
    @{name='left';speed=.5;yaw=.5;push=0.},
    @{name='right';speed=.5;yaw=-.5;push=0.},
    @{name='push05';speed=.5;yaw=0.;push=.5},
    @{name='target10';speed=1.;yaw=0.;push=0.}
)
$taskSummary=[ordered]@{protocol='common_physics_candidate21_screen_v1';status='running';
    checkpoint=$Checkpoint;checkpoint_sha256=(Get-FileHash -LiteralPath $Checkpoint).Hash.ToLowerInvariant();
    source_snapshot=$taskSources;started_at=(Get-Date).ToString('o');rows=@();promotion_performed=$false;
    scope='6 common-physics cases at one deterministic seed, plus15 original-physics old-function cases over3seeds.1.0 is a diagnostic, not fast-running certification. No integrated flow.'}
function Save-CommonScreen {
    [IO.File]::WriteAllText($taskSummaryPath,($taskSummary|ConvertTo-Json -Depth 15),[Text.UTF8Encoding]::new($false))
}
function Assert-ScreenSources {
    foreach($taskItem in $taskSources){
        if((Get-FileHash -LiteralPath $taskItem.path).Hash -ne $taskItem.sha256){throw "Screen source/input drift: $($taskItem.path)"}
    }
}
Save-CommonScreen
try {
    foreach($taskPhysics in @('recovery','original')) {
        $taskSeeds=if($taskPhysics -eq 'recovery'){@(20260909)}else{@(20260909,20260910,20260911)}
        foreach($taskCase in $taskCases) {
            if($taskPhysics -eq 'original' -and $taskCase.name -eq 'target10'){continue}
            foreach($taskSeed in $taskSeeds) {
                Assert-ScreenSources
                $taskName="$taskPhysics-$($taskCase.name)-seed$taskSeed"
                $taskDirectory=Join-Path $taskOutput $taskName
                $taskReused=$taskPhysics -eq 'recovery' -and $taskCase.name -eq 'retained08'
                if(-not $taskReused) {
                    $taskLog=Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' "commonphysics4046-$taskName.log"
                    & $taskEvaluator -Checkpoint $Checkpoint -TrainingResult $TrainingResult -PhysicsMode $taskPhysics `
                        -OutputDir $taskDirectory -WalkSpeed $taskCase.speed -YawRate $taskCase.yaw -PushSpeed $taskCase.push `
                        -Seed $taskSeed -NoVideo *> $taskLog
                    if($LASTEXITCODE -ne 0){throw "Case failed operationally: $taskName"}
                }
                $taskInvocation=Get-Content -Raw -LiteralPath (Join-Path $taskDirectory 'invocation.json')|ConvertFrom-Json
                if($taskInvocation.schema -ne 'trained_common_physics_candidate_invocation_v1' -or
                   $taskInvocation.status -ne 'completed_screening_only' -or $taskInvocation.physics_mode -ne $taskPhysics -or
                   $taskInvocation.seed -ne $taskSeed -or $taskInvocation.checkpoint_sha256 -ne $taskSummary.checkpoint_sha256){throw 'No matching completed actual wrapper invocation'}
                foreach($taskInput in $taskInvocation.source_and_inputs){
                    if((Get-FileHash -LiteralPath $taskInput.path).Hash -ne $taskInput.sha256){throw 'Reused/current case inputs changed'}
                }
                $taskReportPath=Join-Path $taskDirectory 'model_4046_stand_walk_stop.json'
                if((Get-FileHash -LiteralPath $taskReportPath).Hash -ne $taskInvocation.report_sha256){throw 'Report hash mismatch'}
                $taskReport=Get-Content -Raw -LiteralPath $taskReportPath|ConvertFrom-Json
                if($taskReport.protocol_version -ne 'trained_common_physics_candidate_retention_v1' -or
                   $taskReport.retention_experiment.physics_mode -ne $taskPhysics -or $taskReport.seed -ne $taskSeed -or
                   $taskReport.checkpoint_sha256 -ne $taskSummary.checkpoint_sha256 -or
                   $taskReport.protocol.walk_speed -ne $taskCase.speed -or $taskReport.protocol.yaw_rate -ne $taskCase.yaw -or
                   $taskReport.protocol.push_delta_vy -ne $taskCase.push -or $taskReport.retention_passed -isnot [bool]){throw 'Case identity mismatch'}
                if($taskReport.candidate_training.formal_updates_verified -ne 100 -or
                   $taskReport.candidate_training.training_receipt_sha256 -ne (Get-FileHash -LiteralPath $TrainingResult).Hash){throw 'Candidate formal training proof differs'}
                $taskTracePath=Join-Path $taskDirectory 'retention_interface.json'
                $taskGeneratedPath=Join-Path $taskDirectory 'retention_generated.py'
                $taskCsv=Join-Path $taskDirectory 'model_4046_stand_walk_stop.csv'
                if([IO.Path]::GetFullPath($taskReport.retention_experiment.trace_path) -ne $taskTracePath -or
                   (Get-FileHash -LiteralPath $taskTracePath).Hash -ne $taskReport.retention_experiment.trace_sha256 -or
                   (Get-FileHash -LiteralPath $taskGeneratedPath).Hash -ne $taskReport.retention_experiment.generated_sha256 -or
                   [IO.Path]::GetFullPath($taskReport.artifacts.csv) -ne $taskCsv -or
                   @(Import-Csv -LiteralPath $taskCsv).Count -ne 900){throw 'Actual trace/generated/CSV evidence missing or changed'}
                $taskWalk=$taskReport.settled_phase_stats.walk
                $taskFailed=@($taskReport.acceptance.PSObject.Properties|Where-Object {-not $_.Value}|ForEach-Object {$_.Name})
                if($taskReport.retention_acceptance.straight_drift -eq $false){$taskFailed+='straight_drift'}
                $taskSummary.rows+=@{physics=$taskPhysics;case=$taskCase.name;seed=$taskSeed;reused_existing=$taskReused;
                    passed=$taskReport.retention_passed;original_checks=$taskReport.passed;
                    straight_drift=$taskReport.retention_acceptance.straight_drift;
                    failed_criteria=$taskFailed;
                    vx=$taskWalk.vx_b_mean;vy=$taskWalk.vy_b_mean;yaw=$taskWalk.yaw_rate_mean;slip=$taskWalk.contact_slip_mean;
                    report=$taskReportPath;report_sha256=(Get-FileHash -LiteralPath $taskReportPath).Hash.ToLowerInvariant();
                    trace_sha256=$taskReport.retention_experiment.trace_sha256;csv_sha256=(Get-FileHash -LiteralPath $taskCsv).Hash.ToLowerInvariant()}
                Save-CommonScreen
                Write-Output "SCREEN $taskName passed=$($taskReport.retention_passed) vx=$($taskWalk.vx_b_mean) vy=$($taskWalk.vy_b_mean)"
            }
        }
    }
    if($taskSummary.rows.Count -ne 21){throw 'Incomplete21-case screen'}
    Assert-ScreenSources
    $taskSummary.status='completed_screening_only'
} catch {
    $taskSummary.status='error'
    $taskSummary['error']=$_.Exception.Message
    throw
} finally {
    $taskSummary['finished_at']=(Get-Date).ToString('o')
    Save-CommonScreen
}
