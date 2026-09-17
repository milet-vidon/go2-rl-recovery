param(
    [Parameter(Mandatory=$true)][ValidateSet('control','balanced')][string]$Arm,
    [Parameter(Mandatory=$true)][ValidatePattern('^20260917-gait-(control|balanced)-[A-Za-z0-9_-]+$')][string]$RunTag,
    [ValidateSet('smoke','formal')][string]$Mode='smoke',
    [string]$SmokeReceipt,
    [switch]$PreflightOnly
)
# Reuse reviewed single-run machinery, without modifying its original source.
$ErrorActionPreference='Stop'
if(-not $RunTag.StartsWith("20260917-gait-$Arm-")){throw 'Arm/run-tag mismatch'}
$taskTemplate=Join-Path $PSScriptRoot 'run_common_physics_adaptation.ps1'
if((Get-FileHash -LiteralPath $taskTemplate -Algorithm SHA256).Hash -ne '17d185e63d4b5005df7b11fb69d13fdc7a633c063a83957907b9a22f640cb874'){throw 'Frozen launcher changed'}
$taskSource=Get-Content -LiteralPath $taskTemplate -Raw
function Replace-GaitAnchor([string]$Old,[string]$New) {
    if(([regex]::Matches($script:taskSource,[regex]::Escape($Old))).Count -ne 1){throw "Frozen launcher anchor changed: $Old"}
    $script:taskSource=$script:taskSource.Replace($Old,$New)
}
Replace-GaitAnchor '^20260917-commonphysics[A-Za-z0-9_-]+$' "^20260917-gait-$Arm-[A-Za-z0-9_-]+`$"
Replace-GaitAnchor "'train_common_physics_adaptation.py'" "'train_balanced_gait_adaptation.py'"
Replace-GaitAnchor '{2}else{100}' '{2}else{300}'
Replace-GaitAnchor "'bounded_common_physics_adaptation_training_v1'" "'balanced_gait_common_physics_training_v1'"
Replace-GaitAnchor "protocol='bounded_common_physics_launcher_v1'" "protocol='matched_gait_common_physics_launcher_v1';arm='$Arm'"
Replace-GaitAnchor "'--task','Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0'" "'--arm','$Arm','--task','Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0'"
# ScriptBlock has no source file: resolve its two path references, not an automatic variable.
Replace-GaitAnchor "Join-Path `$PSScriptRoot 'train_balanced_gait_adaptation.py'" "Join-Path '$PSScriptRoot' 'train_balanced_gait_adaptation.py'"
Replace-GaitAnchor "Join-Path `$PSScriptRoot 'verify_overnight_baselines.py'" "Join-Path '$PSScriptRoot' 'verify_overnight_baselines.py'"
$taskParameters=@{RunTag=$RunTag;Mode=$Mode;PreflightOnly=$PreflightOnly}
if($SmokeReceipt){$taskParameters.SmokeReceipt=$SmokeReceipt}
& ([scriptblock]::Create($taskSource)) @taskParameters
