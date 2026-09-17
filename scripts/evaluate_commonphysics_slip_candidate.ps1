param(
    [Parameter(Mandatory=$true)][ValidateSet('A','B')][string]$Arm,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$TrainingResult,
    [Parameter(Mandatory=$true)][ValidateSet('original','recovery')][string]$PhysicsMode,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [double]$WalkSpeed=.5,[double]$LateralSpeed=0,[double]$YawRate=0,[double]$PushSpeed=0,
    [int]$Seed=20260909,
    [ValidateSet('legacy','front','oblique')][string]$View='legacy',
    [switch]$NoVideo,[switch]$PreflightOnly,[switch]$SourceOnly
)
# New final4545 identity wrapper around frozen physical900-step evaluator.
# SourceOnly returns text for static review; it executes no Python or simulator.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$taskTemplate=Join-Path $PSScriptRoot 'evaluate_balanced_gait_candidate.ps1'
if((Get-FileHash -LiteralPath $taskTemplate -Algorithm SHA256).Hash -ne '7cb9ba878f6bb7bfb89edd059aac2350453a70e6dbe21ba4090848dbe2e2751c'){throw 'Frozen case wrapper changed'}
$taskSource=Get-Content -Raw -LiteralPath $taskTemplate
function Replace-SlipCase([string]$Old,[string]$New,[int]$Count=1) {
    if(([regex]::Matches($script:taskSource,[regex]::Escape($Old))).Count -ne $Count){throw "Frozen case wrapper anchor changed: $Old"}
    $script:taskSource=$script:taskSource.Replace($Old,$New)
}
Replace-SlipCase "ValidateSet('control','balanced')" "ValidateSet('A','B')"
Replace-SlipCase '4246' '4545' 6
Replace-SlipCase 'trained_balanced_gait_candidate' 'trained_commonphysics_slip_candidate' 3
Replace-SlipCase 'evaluate_balanced_gait_candidate.py' 'evaluate_commonphysics_slip_candidate.py'
Replace-SlipCase 'test_balanced_gait_candidate.py' 'test_commonphysics_slip_candidate.py'
Replace-SlipCase 'train_balanced_gait_adaptation.py' 'train_commonphysics_slip_adaptation.py'
Replace-SlipCase '3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f' '882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc'
Replace-SlipCase '$taskReport.candidate_training.balanced_duration_weight -ne $(if ($Arm -eq ''control'') {0.0} else {-10.0})' @'
$taskReport.candidate_training.balanced_duration_weight -ne -10.0 -or
        $taskReport.candidate_training.foot_slip_weight -ne $(if ($Arm -ceq 'A') {-.5} else {-1.0})
'@
Replace-SlipCase '$taskPaths = @($PSCommandPath,$taskEntry' ('$taskPaths = @('''+$taskTemplate+''',$PSCommandPath,$taskEntry')
# Resolve all script-relative source identities before creating a private block.
$taskSource=$taskSource.Replace('$PSScriptRoot',("'"+$PSScriptRoot+"'"))
$taskSource=$taskSource.Replace('$PSCommandPath',("'"+$PSCommandPath+"'"))
if($SourceOnly){return $taskSource}
$taskParameters=@{Arm=$Arm;Checkpoint=$Checkpoint;TrainingResult=$TrainingResult;PhysicsMode=$PhysicsMode;
    OutputDir=$OutputDir;WalkSpeed=$WalkSpeed;LateralSpeed=$LateralSpeed;YawRate=$YawRate;PushSpeed=$PushSpeed;
    Seed=$Seed;View=$View;NoVideo=$NoVideo;PreflightOnly=$PreflightOnly}
& ([scriptblock]::Create($taskSource)) @taskParameters
