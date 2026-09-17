param(
    [switch]$PreflightOnly,
    [switch]$ExecuteFinitePair
)
# Default is READ-ONLY preflight. Explicit execution runs exactly A50+18, B50+18.
# No second block, conditional extension, model promotion or quality acceptance.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($PreflightOnly -and $ExecuteFinitePair) { throw 'Choose preflight OR explicit finite execution.' }
$portfolio = Split-Path $PSScriptRoot -Parent
$python = 'E:\IsaacLab\env\python.exe'
$runRoot = 'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_flat'
$artifactRoot = 'E:\IsaacLab\artifacts\recovery-20260917'
$parentRun = '2026-09-17_03-10-21_20260917-speedretention128x300'
$parent = Join-Path $runRoot "$parentRun/model_3947.pt"
$parentSha = '3DFBE03E0AC332FEE688C970B420167AFFCDB6E8CACEE16E40FC12120751DF3F'
$parentSummary = Join-Path $portfolio 'evaluations/20260917-speed-retention-control3947/summary.json'
$pairRoot = Join-Path $artifactRoot '20260917-speed-weight-first-pair'
$pairSummaryPath = Join-Path $pairRoot 'summary.json'
$launcher = Join-Path $PSScriptRoot 'run_speed_tracking_weight_block.ps1'
$guard = Join-Path $PSScriptRoot 'check_speed_tracking_weight_block.py'
$evaluator = Join-Path $PSScriptRoot 'evaluate_speed_candidate.ps1'
$preservation = Join-Path $PSScriptRoot 'verify_overnight_baselines.py'
$seeds = @(20260909,20260910,20260911)
$cases = [ordered]@{
    normal=@{vx=.5;yaw=0.;push=0.}; retained08=@{vx=.8;yaw=0.;push=0.};
    target10=@{vx=1.;yaw=0.;push=0.}; left=@{vx=.5;yaw=.5;push=0.};
    right=@{vx=.5;yaw=-.5;push=0.}; push05=@{vx=.5;yaw=0.;push=.5}
}
$smokes = @{
    A=@{run='2026-09-17_04-45-17_20260917-weightA16x2-smoke';weight='1.5';
        model='E2D79840C22A49E0CED4BA7ECD99A4237CB7E601D74C782B88CECB9FFB572A22';
        audit='650A58CBF6E38DF047EE3D177023AB3BCC1DB2DD30F44E501F8CC143C8EEA630';
        log='78E1C9030EFF5B6EB202FB33C3A88B2890CD3798CB44FA0B8CA8B288A8C9D95F';
        invocation='4F00E46DFE82CBA3EBF29A641137BACC936441880B27117B77976D0D962309F7';
        env='407EEB524296CBDBBDB47C8919E78D3F5934CC8221F21AEBF5813EEECE21A3A6';
        agent='561BFD6BAAAE30AF40D74981CF37072E908D0F15E162017BDE12FA2295FDC1BD'};
    B=@{run='2026-09-17_04-45-40_20260917-weightB16x2-smoke';weight='2.0';
        model='B6607B46880A44E2B98056A5A38221571B09572C0DA2437A4055571470BF5AE8';
        audit='267B9D23200F9937D05EF524540348810DA31DF30AC05E8EF310BF5CA56D1972';
        log='8B593C4765C94F03B2DA0A320533CEEB20BD42E28F60843D2ABA9B433FDCACA5';
        invocation='02936AE67D2E2A818DB1512AA695C97FCAC181AC5244E33AFEE3D4274B42CFC8';
        env='1340142BB5F0AA81EC0B557B807579053FBF36388D04101F2BBACE2775033577';
        agent='385AB9E3A7DCE16FA5B0A20FE50CAE5E46F310B2909D8C06AF7E5EC1BABADA97'}
}
$script:frozen = @{}
function Assert-EPath([string]$Path) {
    if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)) -ne 'E:\' -or $Path -match '(^|[/\\])\.\.([/\\]|$)') {
        throw "Expected absolute E: path: $Path"
    }
}
function Add-Frozen([string]$Path,[string]$Hash='') {
    Assert-EPath $Path
    $absolute = [IO.Path]::GetFullPath($Path)
    $actual = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash
    if ($Hash -and $actual -ne $Hash) { throw "Frozen input changed: $absolute" }
    if ($script:frozen.ContainsKey($absolute) -and $script:frozen[$absolute].sha256 -ne $actual) { throw "Input drift: $absolute" }
    $script:frozen[$absolute] = @{path=$absolute;sha256=$actual}
}
function Assert-Sources {
    foreach ($source in $script:frozen.Values) {
        if ((Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash -ne $source.sha256) { throw "Source/input drift: $($source.path)" }
    }
}
function Assert-Idle {
    $busy = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit.*|isaac-sim.*)\.exe$' -and
        ($_.CommandLine -match 'E:[/\\]IsaacLab(?:[/\\]|\b)' -or $_.ExecutablePath -match '^E:[/\\]IsaacLab(?:[/\\]|\b)')
    })
    if ($busy.Count) { throw "Workspace Python/Kit active ($($busy.ProcessId -join ',')); wait, never overlap." }
}
function Assert-HealthyLog([string]$Path) {
    $bad = Select-String -LiteralPath $Path -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|\[Error\].*PhysX|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($null -ne $bad) { throw "Invalid simulation log: $Path : $($bad.Line)" }
}
function Assert-Preservation {
    Assert-Sources
    & $python -I -B $preservation
    if ($LASTEXITCODE -ne 0) { throw '16-model preservation failed.' }
}
function Assert-Number($Value,[string]$Label) {
    if ($null -eq $Value -or $Value -is [string] -or $Value -is [bool] -or
        [double]::IsNaN([double]$Value) -or [double]::IsInfinity([double]$Value)) { throw "Nonfinite/nonnumeric $Label" }
}
function Assert-NewOutputs {
    $paths = @($pairRoot)
    foreach ($arm in @('A','B')) {
        $trainTag = "20260917-weight${arm}128x50-first"
        $evalTag = "20260917-speed-weight${arm}3996"
        $paths += @((Join-Path $artifactRoot $trainTag),(Join-Path $portfolio "configs/$trainTag"),
                    (Join-Path $artifactRoot $evalTag),(Join-Path $portfolio "evaluations/$evalTag"))
        if (@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object {$_.Name.EndsWith("_$trainTag")}).Count) {
            throw "Existing training identity: $trainTag"
        }
    }
    foreach ($path in $paths) { Assert-EPath $path; if (Test-Path -LiteralPath $path) { throw "Preserve existing output: $path" } }
}
function Assert-CheckpointAudit($Audit,[string]$Arm,[int]$Envs,[int]$Updates,[int]$Iteration,[int]$Step,[string]$Hash) {
    if ($Audit.status -ne 'passed' -or $Audit.quality_accepted -ne $false -or $Audit.arm -ne $Arm -or
        $Audit.weight -ne $smokes[$Arm].weight -or $Audit.num_envs -ne $Envs -or $Audit.additional_updates -ne $Updates -or
        $Audit.expected_final_label -ne $Iteration -or $Audit.expected_adam_step -ne $Step -or
        $Audit.parent.sha256 -ne $parentSha -or $Audit.parent.iter -ne 3947 -or
        $Audit.parent.optimizer_group.lr -ne 1e-5 -or @($Audit.parent.adam_steps).Count -ne 17 -or
        @($Audit.parent.adam_steps | Where-Object {$_ -ne 79120}).Count -or
        $Audit.checkpoint.iter -ne $Iteration -or $Audit.checkpoint.sha256 -ne $Hash -or
        $Audit.checkpoint.tensor_count -ne 68 -or $Audit.checkpoint.all_tensors_finite -ne $true -or
        @($Audit.checkpoint.adam_steps).Count -ne 17 -or @($Audit.checkpoint.adam_steps | Where-Object {$_ -ne $Step}).Count) {
        throw "Wrong saved checkpoint/config evidence for arm $Arm"
    }
}
function Read-Screen([string]$Path,[string]$Checkpoint,[string]$Hash) {
    # Validate complete 6x3 identities and actual reports, not just summary flags.
    Assert-EPath $Path
    $s = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $task = 'Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0'
    if ($s.schema_version -ne 'speed_candidate_regression_v1' -or $s.status -ne 'completed_screening_only' -or
        $s.task -ne $task -or $s.checkpoint_sha256 -ne $Hash -or
        [IO.Path]::GetFullPath($s.checkpoint) -ne [IO.Path]::GetFullPath($Checkpoint) -or
        $s.acceptance_eligible -ne $false -or $s.promotion_performed -ne $false -or @($s.rows).Count -ne 18) {
        throw "Incomplete/incorrect 18-case screen: $Path"
    }
    foreach ($source in $s.source_snapshot) { Add-Frozen $source.path $source.sha256 }
    $rows = [Collections.Generic.List[object]]::new()
    $seen = @{}
    foreach ($row in $s.rows) {
        if (-not $cases.Contains($row.case) -or $row.seed -notin $seeds) { throw 'Unknown screen case/seed' }
        $key = "$($row.case):$($row.seed)"
        if ($seen.ContainsKey($key)) { throw "Duplicate screen row: $key" }; $seen[$key]=$true
        $case = $cases[$row.case]
        $expectedReport = Join-Path (Split-Path $Path -Parent) "$($row.case)-seed$($row.seed)/$([IO.Path]::GetFileNameWithoutExtension($Checkpoint))_stand_walk_stop.json"
        if ([IO.Path]::GetFullPath($row.report) -ne [IO.Path]::GetFullPath($expectedReport)) { throw 'Wrong report path' }
        Add-Frozen $row.report $row.report_sha256
        $r = Get-Content -LiteralPath $row.report -Raw | ConvertFrom-Json
        if ($r.protocol_version -ne 'stand_walk_stop_stance_geometry_v2' -or $r.task -ne $task -or
            $r.seed -ne $row.seed -or $r.checkpoint_sha256 -ne $Hash -or $null -ne $r.video_view -or
            [IO.Path]::GetFullPath($r.checkpoint) -ne [IO.Path]::GetFullPath($Checkpoint)) { throw 'Report identity mismatch' }
        if ($r.protocol.walk_speed -ne $case.vx -or $r.protocol.yaw_rate -ne $case.yaw -or
            $r.protocol.push_delta_vy -ne $case.push -or $r.protocol.lateral_speed -ne 0 -or
            $r.protocol.stand_s -ne 4 -or $r.protocol.walk_s -ne 8 -or $r.protocol.stop_s -ne 6 -or
            $row.commanded_vx -ne $case.vx -or $row.commanded_yaw -ne $case.yaw -or $row.push_delta_vy -ne $case.push) {
            throw 'Changed stand/walk/stop command protocol'
        }
        $criteria = @($r.acceptance.PSObject.Properties)
        if (-not $criteria.Count -or @($criteria | Where-Object {$_.Value -isnot [bool]}).Count -or
            $r.passed -isnot [bool] -or $row.report_passed -isnot [bool]) { throw 'Invalid acceptance fields' }
        $failed = @($criteria | Where-Object {-not $_.Value} | ForEach-Object {$_.Name})
        if ($r.passed -ne ($failed.Count -eq 0) -or $row.report_passed -ne $r.passed -or
            (($failed | Sort-Object) -join '|') -ne ((@($row.failed_criteria) | Sort-Object) -join '|')) { throw 'Inconsistent report aggregate' }
        $walk = $r.settled_phase_stats.walk
        $map = @{actual_vx='vx_b_mean';actual_vy='vy_b_mean';actual_yaw='yaw_rate_mean';contact_slip_mean='contact_slip_mean';max_tilt_deg='max_tilt_deg'}
        foreach ($field in $map.Keys) {
            Assert-Number $row.$field "$key summary.$field"
            Assert-Number $walk.($map[$field]) "$key report.$field"
            if ([double]$row.$field -ne [double]$walk.($map[$field])) { throw "Summary/report metric mismatch: $key $field" }
        }
        $straight = $case.yaw -eq 0 -and $case.push -eq 0
        $drift = if ($straight) { [Math]::Abs($walk.vy_b_mean) -lt .12 -and [Math]::Abs($walk.yaw_rate_mean) -lt .15 } else { $null }
        if ($row.straight_drift_passed -ne $drift) { throw 'Inconsistent drift gate' }
        $expectedLog = Join-Path $artifactRoot "$((Split-Path (Split-Path $Path -Parent) -Leaf))/$($row.case)-seed$($row.seed).log"
        if ([IO.Path]::GetFullPath($row.log) -ne [IO.Path]::GetFullPath($expectedLog)) { throw 'Wrong evaluation log path' }
        Assert-HealthyLog $row.log
        Add-Frozen $row.log
        [void]$rows.Add([pscustomobject]@{case=$row.case;seed=$row.seed;report_passed=$r.passed;failed_criteria=$failed;
            actual_vx=[double]$walk.vx_b_mean;actual_vy=[double]$walk.vy_b_mean;actual_yaw=[double]$walk.yaw_rate_mean;
            contact_slip_mean=[double]$walk.contact_slip_mean;straight_drift_passed=$drift;
            target_abs_vx_error=if($row.case -eq 'target10'){[Math]::Abs(1.-$walk.vx_b_mean)}else{$null};
            report=$row.report;report_sha256=$row.report_sha256})
    }
    Add-Frozen $Path
    $target = @($rows | Where-Object case -eq 'target10' | Sort-Object seed)
    return [pscustomobject]@{summary=$Path;summary_sha256=(Get-FileHash -LiteralPath $Path).Hash;
        rows=$rows.ToArray();old_passed=@($rows | Where-Object {$_.case -ne 'target10' -and $_.report_passed}).Count;
        total_passed=@($rows | Where-Object report_passed).Count;
        target_passed=@($target | Where-Object report_passed).Count;
        old_failures=@($rows | Where-Object {$_.case -ne 'target10' -and -not $_.report_passed});
        drift_failures=@($rows | Where-Object {$null -ne $_.straight_drift_passed -and -not $_.straight_drift_passed});
        target_slip_failures=@($target | Where-Object {$_.contact_slip_mean -ge .12});target=$target;
        target_mean_abs_vx_error=($target | Measure-Object target_abs_vx_error -Average).Average}
}
function Judge-Screen($Screen,$Baseline,[string]$Arm) {
    $loss = @($Screen.old_failures).Count -gt 0 -or @($Screen.drift_failures).Count -gt 0 -or @($Screen.target_slip_failures).Count -gt 0
    $deltas = @()
    foreach ($row in $Screen.rows) {
        $base = @($Baseline.rows | Where-Object {$_.case -eq $row.case -and $_.seed -eq $row.seed})[0]
        $deltas += [pscustomobject]@{case=$row.case;seed=$row.seed;delta_vx=$row.actual_vx-$base.actual_vx;
            delta_vy=$row.actual_vy-$base.actual_vy;delta_yaw=$row.actual_yaw-$base.actual_yaw;
            delta_slip=$row.contact_slip_mean-$base.contact_slip_mean}
    }
    $targetChanges = @()
    foreach ($row in $Screen.target) {
        $base = @($Baseline.target | Where-Object seed -eq $row.seed)[0]
        $targetChanges += [pscustomobject]@{seed=$row.seed;candidate_error=$row.target_abs_vx_error;
            parent_error=$base.target_abs_vx_error;worsening=$row.target_abs_vx_error-$base.target_abs_vx_error}
    }
    $improvement = $Baseline.target_mean_abs_vx_error - $Screen.target_mean_abs_vx_error
    $futility = if ($Arm -eq 'B') { $improvement -lt .01 -or @($targetChanges | Where-Object {$_.worsening -gt .02}).Count -gt 0 } else { $null }
    $decision = if ($loss) {'rejected_stop_loss'} elseif ($futility -eq $true) {'rejected_futility'} elseif ($Screen.total_passed -ne 18) {'rejected_target_criteria'} else {'completed_development_screen_only_not_accepted'}
    return [pscustomobject]@{arm=$Arm;decision=$decision;acceptance_eligible=$false;promotion_performed=$false;
        stop_loss=$loss;old_reports_passed=$Screen.old_passed;old_reports_total=15;all_reports_passed=$Screen.total_passed;
        old_failures=$Screen.old_failures;straight_drift_failures=$Screen.drift_failures;target_slip_at_or_above_point12=$Screen.target_slip_failures;
        target_mean_abs_vx_error=$Screen.target_mean_abs_vx_error;target_mean_error_improvement_vs_parent=$improvement;
        b_futility_triggered=$futility;b_futility_required_mean_improvement=.01;b_futility_max_single_seed_worsening=.02;
        target_seed_errors=$targetChanges;continuous_differences_vs_parent=$deltas;screen=$Screen}
}

Assert-Idle
Assert-NewOutputs
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:TEMP='E:\IsaacLab\tmp'
$env:TMP='E:\IsaacLab\tmp'
Add-Frozen $parent $parentSha
Add-Frozen $launcher '34508F025F7274D08715E161F69257FB7DEE2EDDD1EDD4C0D89884B8F447A43D'
Add-Frozen $guard '2FC3D87E69F73ECC6110AA9D8CD396E016A54983F6021016CFCA1850B5FB73F7'
Add-Frozen $preservation '2FBD670D819278CAA19142D77BC65ECC74F960E4AFC19E8142FEE3E507CBA41C'
Add-Frozen (Join-Path $portfolio 'configs/overnight_preservation_20260917.json') '0250A8C60DB31FB21DF03FFD16591D8C8E9BFE61E1C0D85A76E91C2929648E55'
Add-Frozen $parentSummary 'CF33B2809E700D834C195B5DC8317310CCD1BD168F81671729D85C44BFF09379'
Add-Frozen $PSCommandPath
$smokeRecords = @()
foreach ($arm in @('A','B')) {
    $spec = $smokes[$arm]
    $tag = "20260917-weight${arm}16x2-smoke"
    $directory = Join-Path $artifactRoot $tag
    $checkpoint = Join-Path $runRoot "$($spec.run)/model_3948.pt"
    $config = Join-Path $portfolio "configs/$tag"
    Add-Frozen $checkpoint $spec.model
    Add-Frozen (Join-Path $directory 'config-checkpoint-check.json') $spec.audit
    Add-Frozen (Join-Path $directory 'train.log') $spec.log
    Add-Frozen (Join-Path $directory 'invocation.json') $spec.invocation
    Add-Frozen (Join-Path $directory 'source-snapshot.json') '9E571C8022462B8649B4C22D2B473FAE17100DC8C32DAE97B15C768DDF15FC75'
    foreach ($name in @('env','agent')) {
        Add-Frozen (Join-Path $config "$name.yaml") $spec[$name]
        Add-Frozen (Join-Path $runRoot "$($spec.run)/params/$name.yaml") $spec[$name]
    }
    $audit = Get-Content -LiteralPath (Join-Path $directory 'config-checkpoint-check.json') -Raw | ConvertFrom-Json
    Assert-CheckpointAudit $audit $arm 16 2 3948 79160 $spec.model
    foreach ($source in $audit.sources) { Add-Frozen $source.path $source.sha256 }
    Assert-HealthyLog (Join-Path $directory 'train.log')
    Assert-Idle
    $fresh = & $python -B $guard --arm $arm --num-envs 16 --iterations 2 --run-name $tag --candidate-dir $config --checkpoint $checkpoint
    if ($LASTEXITCODE -ne 0) { throw "Fresh full smoke config/checkpoint guard failed: $arm" }
    $freshAudit = ($fresh -join [Environment]::NewLine) | ConvertFrom-Json
    Assert-CheckpointAudit $freshAudit $arm 16 2 3948 79160 $spec.model
    $smokeRecords += @{arm=$arm;checkpoint=$checkpoint;sha256=$spec.model;weight=$spec.weight;final_label=3948;adam_step=79160;finite_tensors=68;config=$config}
}
$baseline = Read-Screen $parentSummary $parent $parentSha
if ($baseline.old_passed -ne 15 -or $baseline.total_passed -ne 15 -or @($baseline.drift_failures).Count -or @($baseline.target_slip_failures).Count) {
    throw 'Frozen control screen no longer has the declared 15 old passes, drift/slip retained.'
}
Assert-Preservation
foreach ($arm in @('A','B')) {
    Assert-Idle
    & $launcher -Arm $arm -RunTag "20260917-weight${arm}128x50-first" -NumEnvs 128 -Iterations 50 -PreflightOnly
}
Assert-Sources
Assert-Preservation
Assert-Idle
Write-Output 'Verified both real16x2 smokes, frozen control3947, complete parent18 reports, all sources and 16 preserved models.'
if (-not $ExecuteFinitePair) {
    Write-Output 'READ-ONLY preflight complete. No outputs/simulator created. Explicit -ExecuteFinitePair is required for A50+18 then B50+18.'
    return
}

Assert-NewOutputs
New-Item -ItemType Directory -Path $pairRoot | Out-Null
$script:frozen.Values | Sort-Object path | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $pairRoot 'source-snapshot.json') -Encoding UTF8
$summary = [ordered]@{schema_version='speed_tracking_first_pair_diagnostic_v1';status='running';started_at=(Get-Date -Format o);
    acceptance_eligible=$false;promotion_performed=$false;automatic_extension=$false;blocks_per_arm=1;
    parent_checkpoint=$parent;parent_sha256=$parentSha;smokes=$smokeRecords;baseline=$baseline;arms=@();
    execution_order='A:128envx50update,18freshcases; B:independent same parent128envx50update,18freshcases';
    quality_failure_policy='Reject that arm, preserve all evidence, compare the other independent arm. Execution/integrity errors stop the whole pair.';
    limitations='Engineering diagnostic only; evaluated development seeds are reused. No animal imitation, integrated recovery, hardware or reference-video acceptance.'}
function Save-Summary { $summary | ConvertTo-Json -Depth 25 | Set-Content -LiteralPath $pairSummaryPath -Encoding UTF8 }
Save-Summary
try {
    foreach ($arm in @('A','B')) {
        $trainTag = "20260917-weight${arm}128x50-first"
        $evalTag = "20260917-speed-weight${arm}3996"
        Assert-Idle
        Assert-Preservation
        & $launcher -Arm $arm -RunTag $trainTag -NumEnvs 128 -Iterations 50 *> (Join-Path $pairRoot "arm$arm-training.log")
        Assert-Idle
        Assert-Preservation
        $runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object {$_.Name.EndsWith("_$trainTag")})
        if ($runs.Count -ne 1) { throw "Expected one arm$arm finite run" }
        $checkpoint = Join-Path $runs[0].FullName 'model_3996.pt'
        $hash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash
        $auditPath = Join-Path $artifactRoot "$trainTag/config-checkpoint-check.json"
        $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
        Assert-CheckpointAudit $audit $arm 128 50 3996 80120 $hash
        Add-Frozen $checkpoint $hash
        Add-Frozen $auditPath
        foreach ($name in @('env.yaml','agent.yaml')) {
            $archived = Join-Path $portfolio "configs/$trainTag/$name"
            Add-Frozen $archived
            Add-Frozen (Join-Path $runs[0].FullName "params/$name") (Get-FileHash -LiteralPath $archived).Hash
        }
        Assert-HealthyLog (Join-Path $artifactRoot "$trainTag/train.log")
        Add-Frozen (Join-Path $artifactRoot "$trainTag/train.log")
        $record = [ordered]@{arm=$arm;status='trained_waiting_for18';checkpoint=$checkpoint;checkpoint_sha256=$hash;
            final_label=3996;adam_step=80120;guard=$auditPath;decision=$null}
        $summary.arms += $record
        Save-Summary
        Assert-Idle
        Assert-Preservation
        & $evaluator -Checkpoint $checkpoint -RunTag $evalTag *> (Join-Path $pairRoot "arm$arm-regression.log")
        Assert-Idle
        Assert-Preservation
        $screen = Read-Screen (Join-Path $portfolio "evaluations/$evalTag/summary.json") $checkpoint $hash
        $record.decision = Judge-Screen $screen $baseline $arm
        $record.status = 'completed_diagnostic_only'
        Save-Summary
        Write-Output "Arm $arm : $($record.decision.decision), old=$($screen.old_passed)/15, total=$($screen.total_passed)/18. No acceptance/extension."
    }
    $a = $summary.arms[0].decision
    $b = $summary.arms[1].decision
    $summary.matched_budget_comparison = @{b_target_mean_error_improvement_vs_a=$a.target_mean_abs_vx_error-$b.target_mean_abs_vx_error;
        a_decision=$a.decision;b_decision=$b.decision;causal_replication_proven=$false;promotion_performed=$false}
    $summary.status='completed_diagnostics_only'
} catch {
    $summary.status='error'
    $summary.error=$_.Exception.Message
    throw
} finally {
    try {
        Assert-Preservation
        $summary.integrity_after_pair='passed'
    } catch {
        $summary.status='error'
        $summary.integrity_after_pair='failed'
        $summary.integrity_error=$_.Exception.Message
        throw
    } finally {
        $summary.finished_at=Get-Date -Format o
        $script:frozen.Values | Sort-Object path | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $pairRoot 'final-source-and-evidence-snapshot.json') -Encoding UTF8
        Save-Summary
    }
}
Write-Output "Finite pair completed (not accepted): $pairSummaryPath"
