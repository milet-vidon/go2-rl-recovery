param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][ValidatePattern('^20260917-speed-[A-Za-z0-9_-]+$')][string]$RunTag,
    [switch]$PreflightOnly
)
# Bounded post-training regressions; never promotes a model or changes acceptance.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$portfolio = Split-Path $PSScriptRoot -Parent
$python = 'E:\IsaacLab\env\python.exe'
$task = 'Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0'
$out = Join-Path $portfolio "evaluations/$RunTag"
$logs = Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
$evaluator = Join-Path $PSScriptRoot 'evaluate_stand_walk_stop.ps1'
$verifier = Join-Path $PSScriptRoot 'verify_overnight_baselines.py'
$seeds = @(20260909,20260910,20260911)
$cases = @(
    @{name='normal'; speed=.5; yaw=0.; push=0.},
    @{name='retained08'; speed=.8; yaw=0.; push=0.},
    @{name='target10'; speed=1.; yaw=0.; push=0.},
    @{name='left'; speed=.5; yaw=.5; push=0.},
    @{name='right'; speed=.5; yaw=-.5; push=0.},
    @{name='push05'; speed=.5; yaw=0.; push=.5}
)
foreach($path in @($Checkpoint,$out,$logs,$portfolio,$python)) {
    if([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($path)) -ne 'E:\'){throw "Non-E path: $path"}
}
if(-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)){throw 'Missing candidate checkpoint'}
foreach($path in @($out,$logs)){if(Test-Path -LiteralPath $path){throw "Preserve existing output: $path"}}
$sha=(Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash
$installed='E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$sourcePaths=@($PSCommandPath,$evaluator,$verifier,(Join-Path $PSScriptRoot 'evaluate_go2_stand_walk_stop.py'))
$sourcePaths+=@(Get-ChildItem -LiteralPath $installed -Recurse -File -Filter '*.py' | ForEach-Object {$_.FullName})
$sources=@($sourcePaths | ForEach-Object {@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash}})
function Assert-Idle {
    $active=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pythonw.exe','kit.exe','isaac-sim.exe') -and $_.CommandLine -match 'E:[/\\]IsaacLab' -and
        ($_.Name -in @('kit.exe','isaac-sim.exe') -or $_.CommandLine -match 'rsl_rl[/\\]train.py|resume_smith_checkpoint_entry.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py')
    })
    if($active.Count){throw "Another simulator active: $($active.ProcessId -join ',')"}
}
function Assert-Integrity {
    & $python -B $verifier
    if($LASTEXITCODE -ne 0){throw 'Frozen baselines changed'}
    if((Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash -ne $sha){throw 'Candidate changed during evaluation'}
    foreach($source in $sources){if((Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash -ne $source.sha256){throw "Source changed: $($source.path)"}}
}
Assert-Idle
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP='E:\IsaacLab\tmp'
$env:TMP='E:\IsaacLab\tmp'
Assert-Integrity
if($PreflightOnly){Write-Output '18-case regression preflight passed; no simulator/output created.';return}
New-Item -ItemType Directory -Path $out,$logs | Out-Null
$summary=[ordered]@{schema_version='speed_candidate_regression_v1'; status='running'; started_at=(Get-Date -Format o); task=$task;checkpoint=$Checkpoint;checkpoint_sha256=$sha; source_snapshot=$sources; rows=@(); promotion_performed=$false; acceptance_eligible=$false; limitations='Development simulation screening only. No high-speed or integrated recovery acceptance. No videos in this batch; turn/push controls require matched frozen-baseline comparison before a no-regression claim.'}
$summaryPath=Join-Path $out 'summary.json'
function Save-Summary {$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryPath -Encoding UTF8}
Save-Summary
try {
    foreach($case in $cases){foreach($seed in $seeds){
        Assert-Idle
        Assert-Integrity
        $name="$($case.name)-seed$seed"
        $directory=Join-Path $out $name
        $log=Join-Path $logs "$name.log"
        Write-Output "$(Get-Date -Format o) Starting $name"
        & $evaluator -Checkpoint $Checkpoint -Task $task -OutputDir $directory -Seed $seed -WalkSpeed $case.speed -YawRate $case.yaw -PushSpeed $case.push -LateralSpeed 0 -NoVideo *> $log
        if($LASTEXITCODE -ne 0){throw "Evaluation failed: $log"}
        $bad=Select-String -LiteralPath $log -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|\[Error\].*PhysX|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
        if($null -ne $bad){throw "Invalid simulation: $($bad.Line)"}
        Assert-Integrity
        $reportPath=Join-Path $directory (([IO.Path]::GetFileNameWithoutExtension($Checkpoint))+'_stand_walk_stop.json')
        $r=Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if($r.protocol_version -ne 'stand_walk_stop_stance_geometry_v2' -or $r.task -ne $task -or $r.seed -ne $seed -or $r.checkpoint_sha256 -ne $sha -or $r.passed -isnot [bool] -or $null -ne $r.video_view){throw 'Wrong candidate report identity'}
        if([IO.Path]::GetFullPath($r.checkpoint) -ne [IO.Path]::GetFullPath($Checkpoint)){throw 'Wrong checkpoint path'}
        if($r.protocol.walk_speed -ne $case.speed -or $r.protocol.yaw_rate -ne $case.yaw -or $r.protocol.push_delta_vy -ne $case.push -or $r.protocol.lateral_speed -ne 0 -or $r.protocol.stand_s -ne 4 -or $r.protocol.walk_s -ne 8 -or $r.protocol.stop_s -ne 6){throw 'Wrong command protocol'}
        $acceptance=@($r.acceptance.PSObject.Properties)
        if(-not $acceptance.Count -or @($acceptance | Where-Object {$_.Value -isnot [bool]}).Count){throw 'Invalid acceptance fields'}
        $failed=@($acceptance | Where-Object {-not $_.Value} | ForEach-Object {$_.Name})
        if($r.passed -ne ($failed.Count -eq 0)){throw 'Inconsistent acceptance flag'}
        $walk=$r.settled_phase_stats.walk
        foreach($field in @('vx_b_mean','vy_b_mean','yaw_rate_mean','contact_slip_mean','max_tilt_deg')){
            $value=$walk.$field
            if($null -eq $value -or $value -is [string] -or $value -is [bool] -or [double]::IsNaN([double]$value) -or [double]::IsInfinity([double]$value)){throw "Invalid walk metric: $field"}
        }
        # New straightness gates apply to straight, unperturbed rollouts only.
        $straight=($case.yaw -eq 0 -and $case.push -eq 0)
        $drift=if($straight){[Math]::Abs($walk.vy_b_mean) -lt .12 -and [Math]::Abs($walk.yaw_rate_mean) -lt .15}else{$null}
        $summary.rows+=@{case=$case.name;seed=$seed;commanded_vx=$case.speed;commanded_yaw=$case.yaw;push_delta_vy=$case.push;actual_vx=$walk.vx_b_mean;actual_vy=$walk.vy_b_mean;actual_yaw=$walk.yaw_rate_mean;contact_slip_mean=$walk.contact_slip_mean;max_tilt_deg=$walk.max_tilt_deg;report_passed=$r.passed;failed_criteria=$failed;straight_drift_passed=$drift;report=$reportPath;report_sha256=(Get-FileHash -LiteralPath $reportPath).Hash;log=$log}
        Save-Summary
        Write-Output "$name passed=$($r.passed), vx=$($walk.vx_b_mean), slip=$($walk.contact_slip_mean)"
    }}
    $summary.status='completed_screening_only'
} catch {
    $summary.status='error'
    $summary.error=$_.Exception.Message
    throw
} finally {
    $summary.finished_at=Get-Date -Format o
    Save-Summary
}
Write-Output "Finite18-case screen completed: $summaryPath. No promotion."
