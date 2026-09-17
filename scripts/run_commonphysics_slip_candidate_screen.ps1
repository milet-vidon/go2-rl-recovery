param(
    [Parameter(Mandatory=$true)][ValidateSet('A','B')][string]$Arm,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$TrainingResult,
    [switch]$SourceOnly
)
# Predeclared finite33 new cases/arm, never old21 reuse or model promotion.
# Common6 x seeds09/10/11=18; original retained5 x same3=15.
# Byte-identical trajectories are reported as one identity, not3 independent
# trials. Seed labels alone are not evidence of independent generalization.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$taskTemplate=Join-Path $PSScriptRoot 'run_balanced_gait_candidate_screen.ps1'
if((Get-FileHash -LiteralPath $taskTemplate -Algorithm SHA256).Hash -ne '9ed912aa5656fd3e7873b1141fb4c9d73fc8d3b6f3256191e729e1860fe9c9f3'){throw 'Frozen33-case screen base changed'}
$taskSource=Get-Content -Raw -LiteralPath $taskTemplate
function Replace-SlipScreen([string]$Old,[string]$New,[int]$Count=1) {
    if(([regex]::Matches($script:taskSource,[regex]::Escape($Old))).Count -ne $Count){throw "Frozen screen anchor changed: $Old"}
    $script:taskSource=$script:taskSource.Replace($Old,$New)
}
Replace-SlipScreen "ValidateSet('control','balanced')" "ValidateSet('A','B')"
Replace-SlipScreen '$Arm=$Arm.ToLowerInvariant()' '# Arm identity remains exactly A/B.'
Replace-SlipScreen '4246' '4545' 7
Replace-SlipScreen '85120' '91120'
Replace-SlipScreen 'trained_balanced_gait_candidate' 'trained_commonphysics_slip_candidate' 3
Replace-SlipScreen 'evaluate_balanced_gait_candidate' 'evaluate_commonphysics_slip_candidate' 2
Replace-SlipScreen 'test_balanced_gait_candidate' 'test_commonphysics_slip_candidate'
Replace-SlipScreen 'train_balanced_gait_adaptation' 'train_commonphysics_slip_adaptation'
Replace-SlipScreen 'balanced_gait_common_physics_training_v1' 'matched_commonphysics_slip_training_v1'
Replace-SlipScreen 'balanced_gait_candidate21_screen_v1' 'commonphysics_slip_candidate33_screen_v1'
Replace-SlipScreen '3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f' '882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc'
Replace-SlipScreen '20260917-gait-$Arm-4545-first' '20260917-slip-$Arm-4545-first'
Replace-SlipScreen '$taskWeight=if($Arm -eq ''control''){0.0}else{-10.0}' '$taskWeight=-10.0; $taskSlipWeight=if($Arm -ceq ''A''){-.5}else{-1.0}'
Replace-SlipScreen '$taskReceipt.arm -cne $Arm -or $taskReceipt.balanced_duration_weight -ne $taskWeight -or' '$taskReceipt.arm -cne $Arm -or $taskReceipt.balanced_duration_weight -ne $taskWeight -or $taskReceipt.foot_slip_weight -ne $taskSlipWeight -or'
Replace-SlipScreen '$taskProof.arm -cne $Arm -or $taskProof.balanced_duration_weight -ne $taskWeight -or' '$taskProof.arm -cne $Arm -or $taskProof.balanced_duration_weight -ne $taskWeight -or $taskProof.foot_slip_weight -ne $taskSlipWeight -or'
Replace-SlipScreen '$taskSeeds=if($taskPhysics -eq ''recovery''){@(20260909)}else{@(20260909,20260910,20260911)}' '$taskSeeds=@(20260909,20260910,20260911)'
Replace-SlipScreen 'One finite21-case final4545 screen.' 'One finite33-case final4545 screen.'
Replace-SlipScreen '6 newly executed common-physics cases at seed20260909, plus15 newly executed original-physics old-function cases over3seeds. Same21-case protocol as4046.1.0 is diagnostic, not fast-running certification. No integrated flow.' 'Predeclared18 common-physics cases (6 commands x seeds20260909/10/11) and15 original-physics cases (5 retained commands x same3seeds). All newly executed; identical trajectories are not independent trials. Same physical thresholds. Research screening only: no natural-gait, fast-run or integrated-flow acceptance.'
Replace-SlipScreen '$taskSummary.rows.Count -ne 21' '$taskSummary.rows.Count -ne 33'
Replace-SlipScreen 'Incomplete21-case screen.' 'Incomplete33-case screen.'
Replace-SlipScreen "['all21_passed']" "['all33_passed']"
Replace-SlipScreen '$taskSourcePaths=@($PSCommandPath,$taskEvaluator,$taskEntry,' ('$taskSourcePaths=@('''+$taskTemplate+''',$PSCommandPath,$taskEvaluator,$taskEntry,')
Replace-SlipScreen '$taskSummary[''current_case'']=$null' @'
    $taskSummary['trajectory_identity_groups']=@($taskSummary.rows | Group-Object physics,case | ForEach-Object {
        @{physics=$_.Group[0].physics;case=$_.Group[0].case;scheduled_seeds=@($_.Group.seed);
          unique_csv_sha256=@($_.Group.csv_sha256 | Sort-Object -Unique);
          distinct_observed_trajectories=@($_.Group.csv_sha256 | Sort-Object -Unique).Count;
          independent_generalization_claimed=$false}
    })
    $taskSummary['common18_passed']=@($taskSummary.rows|Where-Object {$_.physics -eq 'recovery'}).Count -eq 18 -and @($taskSummary.rows|Where-Object {$_.physics -eq 'recovery' -and -not $_.passed}).Count -eq 0
    $taskSummary['original15_retained']=@($taskSummary.rows|Where-Object {$_.physics -eq 'original'}).Count -eq 15 -and @($taskSummary.rows|Where-Object {$_.physics -eq 'original' -and -not $_.passed}).Count -eq 0
    $taskSummary['quality_accepted']=$false
    $taskSummary['full_flow_passed']=$false
    $taskSummary['natural_gait_accepted']=$false
    $taskSummary['current_case']=$null
'@
$taskSource=$taskSource.Replace('$PSScriptRoot',("'"+$PSScriptRoot+"'"))
$taskSource=$taskSource.Replace('$PSCommandPath',("'"+$PSCommandPath+"'"))
if($SourceOnly){return $taskSource}
& ([scriptblock]::Create($taskSource)) -Arm $Arm -Checkpoint $Checkpoint -TrainingResult $TrainingResult
