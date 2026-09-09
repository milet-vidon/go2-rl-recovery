param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [int]$Seed = 20260909,
    [string[]]$Cases = @('normal', 'left', 'right', 'fast', 'push'),
    [switch]$Video
)
$ErrorActionPreference = 'Stop'
$scenarios = @{
    normal = @{WalkSpeed=0.5; YawRate=0; PushSpeed=0}
    left = @{WalkSpeed=0.5; YawRate=0.5; PushSpeed=0}
    right = @{WalkSpeed=0.5; YawRate=-0.5; PushSpeed=0}
    fast = @{WalkSpeed=0.8; YawRate=0; PushSpeed=0}
    push = @{WalkSpeed=0.5; YawRate=0; PushSpeed=0.75}
}
foreach ($case in $Cases) { if (-not $scenarios.ContainsKey($case)) { throw "Unknown scenario $case" } }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
foreach ($case in $Cases) {
    $parameters = $scenarios[$case]
    $caseDir = Join-Path $OutputDir $case
    if (Test-Path $caseDir) { throw "Use a fresh output directory: $caseDir" }
    & (Join-Path $PSScriptRoot 'evaluate_stand_walk_stop.ps1') -Checkpoint $Checkpoint -Seed $Seed -NoVideo:(!$Video) -OutputDir $caseDir @parameters *> (Join-Path $OutputDir "$case.log")
    $reportPath = Join-Path $caseDir (([IO.Path]::GetFileNameWithoutExtension($Checkpoint)) + '_stand_walk_stop.json')
    if (-not (Test-Path $reportPath)) { throw "No report for $case. See $OutputDir\$case.log" }
    $report = Get-Content -Raw $reportPath | ConvertFrom-Json
    Write-Host "$case : passed=$($report.passed) tilt=$($report.settled_phase_stats.walk.max_tilt_deg) vx=$($report.settled_phase_stats.walk.vx_b_mean) yaw=$($report.settled_phase_stats.walk.yaw_rate_mean)"
}
