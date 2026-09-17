param(
    [Parameter(Mandatory=$true)][ValidateSet('A','B')][string]$Arm,
    [Parameter(Mandatory=$true)][ValidatePattern('^20260917-slip-(A|B)-[A-Za-z0-9_-]+$')][string]$RunTag,
    [ValidateSet('smoke','formal')][string]$Mode='smoke',
    [string]$SmokeReceipt,
    [string]$ContinuousReport,
    [switch]$PreflightOnly
)
# Does not unlock formal training: Python requires a separately reviewed full
# continuous v3 sensor-identity guard. No old21-case or video-pair success substitute.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $RunTag.StartsWith("20260917-slip-$Arm-")){throw 'Arm/run-tag mismatch'}
if($Mode -eq 'formal' -and -not $ContinuousReport){throw 'Formal requires -ContinuousReport from a newly passed real continuous .5 diagnostic'}
$taskTemplate=Join-Path $PSScriptRoot 'run_common_physics_adaptation.ps1'
if((Get-FileHash -LiteralPath $taskTemplate -Algorithm SHA256).Hash -ne '17d185e63d4b5005df7b11fb69d13fdc7a633c063a83957907b9a22f640cb874'){throw 'Frozen launcher changed'}
$taskSource=Get-Content -LiteralPath $taskTemplate -Raw
$taskParentReceipt='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_flat\2026-09-17_16-33-35_20260917-gait-balanced-128x300-first\common_physics_training_result.json'
if((Get-FileHash -LiteralPath $taskParentReceipt -Algorithm SHA256).Hash -ne '299b45ea7c89e10107c4273e1d4867e546c3017bb8e18ee48cc9664427c91b9e'){throw 'Frozen parent receipt changed'}
$taskParentState=Get-Content -LiteralPath $taskParentReceipt -Raw | ConvertFrom-Json
$taskParentRate=([double]$taskParentState.checkpoint_metadata.optimizer_group.lr).ToString('R',[Globalization.CultureInfo]::InvariantCulture)
function Replace-SlipAnchor([string]$Old,[string]$New) {
    if(([regex]::Matches($script:taskSource,[regex]::Escape($Old))).Count -ne 1){throw "Frozen launcher anchor changed: $Old"}
    $script:taskSource=$script:taskSource.Replace($Old,$New)
}
Replace-SlipAnchor '^20260917-commonphysics[A-Za-z0-9_-]+$' "^20260917-slip-$Arm-[A-Za-z0-9_-]+`$"
Replace-SlipAnchor "'train_common_physics_adaptation.py'" "'train_commonphysics_slip_adaptation.py'"
Replace-SlipAnchor "'2026-09-17_03-10-21_20260917-speedretention128x300'" "'2026-09-17_16-33-35_20260917-gait-balanced-128x300-first'"
Replace-SlipAnchor "'3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f'" "'882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc'"
Replace-SlipAnchor '{2}else{100}' '{2}else{300}'
Replace-SlipAnchor '$taskExpectedIteration=3947+$taskUpdates-1' '$taskExpectedIteration=4246+$taskUpdates-1'
Replace-SlipAnchor '$taskExpectedAdam=79120+20*$taskUpdates' '$taskExpectedAdam=85120+20*$taskUpdates'
Replace-SlipAnchor "'bounded_common_physics_adaptation_training_v1'" "'matched_commonphysics_slip_training_v1'"
Replace-SlipAnchor "protocol='bounded_common_physics_launcher_v1'" "protocol='matched_commonphysics_slip_launcher_v1';arm='$Arm'"
Replace-SlipAnchor "'--task','Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0'" "'--arm','$Arm','--task','Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0'"
Replace-SlipAnchor "'--checkpoint','model_3947.pt','agent.algorithm.learning_rate=1e-5'" "'--checkpoint','model_4246.pt','agent.algorithm.learning_rate=$taskParentRate'"
Replace-SlipAnchor "Join-Path `$PSScriptRoot 'train_commonphysics_slip_adaptation.py'" "Join-Path '$PSScriptRoot' 'train_commonphysics_slip_adaptation.py'"
Replace-SlipAnchor "Join-Path `$PSScriptRoot 'verify_overnight_baselines.py'" "Join-Path '$PSScriptRoot' 'verify_overnight_baselines.py'"
Replace-SlipAnchor "Push-Location `$taskRepo" "if(`$ContinuousReport){`$taskArgs+=@('--continuous_report',`$ContinuousReport)}`nPush-Location `$taskRepo"
Replace-SlipAnchor 'Training only. No further run, promotion, fast-running or integration claim. Evaluate both recovery and original physics.' 'Research training only from rejected broader-gait balanced4246. No automatic continuation/promotion, natural gait or fast-run claim; new full behavior and continuous tests remain required.'
# Add parameter to the private scriptblock; it does not inherit user input via eval.
Replace-SlipAnchor '    [string]$SmokeReceipt,' "    [string]`$SmokeReceipt,`n    [string]`$ContinuousReport,"
$taskParameters=@{RunTag=$RunTag;Mode=$Mode;PreflightOnly=$PreflightOnly;ContinuousReport=$ContinuousReport}
if($SmokeReceipt){$taskParameters.SmokeReceipt=$SmokeReceipt}
& ([scriptblock]::Create($taskSource)) @taskParameters
