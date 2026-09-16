param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag = '20260916-smithnominal128x2000',
    [ValidateRange(100,3000)][int]$Iterations = 2000,
    [switch]$PreflightOnly
)
# Four serial simulator calls, six raw videos, three paired diagnostics. No model promotion.
$ErrorActionPreference = 'Stop'
$portfolio = Split-Path $PSScriptRoot -Parent
$artifactRoot = 'E:\IsaacLab\artifacts\recovery-20260916'
$runRoot = 'E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery'
$installedRoot = 'E:\IsaacLab\repo\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
$bankPath = Join-Path $portfolio 'datasets/recovery_states/nominal_pd_v1_20260916'
$bankSha = '71B882E92035B4C248E5980339C980DBB242DCB1EC711C05252C33249DB48F5A'
$checkpointName = "model_$($Iterations - 1).pt"
$task = 'Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0'
$output = Join-Path $portfolio "evaluations/$RunTag-video-review"
$python = 'E:\IsaacLab\env\python.exe'
$criterion = 'gravity error < 0.35, height 0.30-0.55 m, speed < 0.50 m/s, angular speed < 1.00 rad/s, 4 simultaneous foot vertical forces > 5 N, no base contact, feet on correct body sides (0.06 < signed lateral < 0.30 m), knees on correct sides (>0.04 m), fore/hind feet on correct ends (>0.08 m), each joint offset <0.65 rad; all continuously held for hold_s'
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($output)) -ne 'E:\') { throw 'Outputs must remain on E:.' }
if (Test-Path -LiteralPath $output) { throw "Existing output: $output. Inspect and resume missing stages manually; never overwrite." }

function Get-ActiveSimulators {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pythonw.exe','kit.exe','isaac-sim.exe') -and
        $_.CommandLine -match 'E:[/\\]IsaacLab' -and
        ($_.Name -in @('kit.exe','isaac-sim.exe') -or
         $_.CommandLine -match 'rsl_rl[/\\]train.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py')
    })
}
function Assert-HealthyLog([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing simulation log: $Path" }
    $bad = Select-String -LiteralPath $Path -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($null -ne $bad) { throw "Invalid simulation: $Path : $($bad.Line)" }
}
function Assert-Report($Report, [string]$Protocol, [int]$Trials, $View = $null) {
    $expectedPoses = switch ($Protocol) {
        'heldout' { @('side','upside_down') }
        'angle30' { @('upright','side','fore_aft') }
        'angle45' { @('upright','side','fore_aft','upside_down') }
        'upright' { @('upright') }
        default { throw "Unknown protocol $Protocol" }
    }
    $expectedSeed = if ($Protocol -eq 'angle45') { 20260916 } else { 20260918 }
    if ($Report.task -ne $task -or $Report.checkpoint_sha256 -ne $modelSha -or
        [IO.Path]::GetFullPath($Report.checkpoint) -ne $checkpoint -or
        $Report.seed -ne $expectedSeed -or $Report.acceptance_eligible -ne $true -or
        $Report.policy_action_mode -ne 'deterministic_mean' -or $Report.self_collisions_enabled -ne $true -or
        $Report.criterion -cne $criterion -or $Report.video_view -ne $View) { throw "Invalid $Protocol evaluation identity/acceptance metadata" }
    $action = $Report.action_representation
    if ($action.reference -ne 'nominal' -or $action.scale -ne 0.25 -or $action.sample_period_s -ne 0.02 -or
        $action.held_over_physics_substeps -ne 4 -or $action.target -ne 'q_reference + scale * action, soft-joint-limit clamped') { throw 'Action representation drift' }
    $poses = @($Report.results.PSObject.Properties.Name)
    if ((($poses | Sort-Object) -join ',') -ne (($expectedPoses | Sort-Object) -join ',')) { throw "Wrong pose set for $Protocol" }
    if ($Protocol -eq 'heldout') {
        if ($Report.protocol_version -ne 'stance_geometry_v1_state_bank_PD_v1' -or
            $Report.start_protocol_id -ne 'state_bank_heldout_nominal_pose_PD' -or
            $Report.settle_requested_s -ne 1 -or $Report.settle_actual_s -ne 1 -or $Report.settle_control_steps -ne 50 -or
            $Report.settle_controller -ne 'direct nominal-position PD, not policy zero action; not zero torque' -or
            $action.pre_policy_handover -ne 'direct nominal-position PD, not policy zero action' -or $null -ne $Report.angle_deg -or
            $Report.state_bank.sha256 -ne $bankSha -or $Report.state_bank.schema_version -ne 'nominal_pd_fallen_v1' -or
            $Report.state_bank.split -ne 'heldout' -or $Report.state_bank.diagnostic_train_split_only -ne $false -or
            [IO.Path]::GetFullPath($Report.state_bank.path) -ne (Join-Path $bankPath 'states.npz')) { throw 'Heldout bank/handover protocol drift' }
    } else {
        $angle = if ($Protocol -eq 'angle45') { 45 } else { 30 }
        if ($Report.protocol_version -ne 'stance_geometry_v1' -or $Report.start_protocol_id -ne 'controlled_drop' -or
            $Report.angle_deg -ne $angle -or $Report.settle_requested_s -ne 0 -or $Report.settle_actual_s -ne 0 -or
            $null -ne $Report.state_bank -or $null -ne $action.pre_policy_handover) { throw 'Controlled-drop protocol drift' }
    }
    $allIds = @()
    foreach ($pose in $expectedPoses) {
        $r = $Report.results.$pose
        if ($r.trials -ne $Trials -or $r.stable_hold_s -ne 3 -or $r.horizon_s -ne 8 -or
            @($r.policy_start_state).Count -ne $Trials -or @($r.final_diagnostics).Count -ne $Trials) { throw "Incomplete $Protocol/$pose trials or changed hold/horizon" }
        foreach ($name in @('successes','final_valid_stands','final_geometry_passes')) {
            if ($name -notin $r.PSObject.Properties.Name -or $r.$name -lt 0 -or $r.$name -gt $Trials) { throw "Invalid outcome $Protocol/$pose/$name" }
        }
        if ($Protocol -eq 'heldout') {
            $ids = @($r.state_bank_selection.selected_state_ids)
            if ($r.settled_fallen_trials -ne $Trials -or $ids.Count -ne $Trials -or
                @($ids | Select-Object -Unique).Count -ne $r.state_bank_selection.selected_unique_state_count -or
                (@($r.policy_start_state.bank_state_id) -join ',') -ne ($ids -join ',') -or
                @($r.policy_start_state | Where-Object { $_.eligible_settled_fallen_recovery -ne $true }).Count) { throw "Invalid settled-fallen sources for $pose" }
            $allIds += $ids
        }
    }
    if ($Protocol -eq 'heldout' -and ($allIds -join ',') -ne (@($Report.state_bank.selected_state_ids) -join ',')) { throw 'Global/per-pose bank source order mismatch' }
}

$reportPaths = @{}
$pending = @()
foreach ($protocol in @('heldout','angle30','angle45')) {
    $path = Join-Path $portfolio "evaluations/$RunTag-$protocol/$($checkpointName)_recovery_metrics.json"
    $reportPaths[$protocol] = $path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $pending += $path }
}
$active = @(Get-ActiveSimulators)
if ($pending.Count -or $active.Count) {
    Write-Output "NOT READY: $($pending.Count) missing completed 20-trial reports; active simulator PIDs: $($active.ProcessId -join ','). No files created or simulator started."
    if ($PreflightOnly) { return }
    throw 'Wait for training and all three serial evaluation suites to finish.'
}
$runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") })
if ($runs.Count -ne 1) { throw 'Expected exactly one matching completed training run.' }
$checkpoint = [IO.Path]::GetFullPath((Join-Path $runs[0].FullName $checkpointName))
$modelSha = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash
$snapshot = @(Get-Content -LiteralPath (Join-Path $artifactRoot "$RunTag-source-snapshot.json") -Raw | ConvertFrom-Json)
$expectedSources = @('recovery_smith_math.py','recovery_smith_mdp.py','recovery_env_cfg.py','recovery_bank_mdp.py',
    'recovery_state_bank.py','recovery_math.py','recovery_mdp.py','recovery_control_targets.py','flat_env_cfg.py','__init__.py','agents/rsl_rl_ppo_cfg.py')
if ($snapshot.Count -ne $expectedSources.Count) { throw 'Incomplete source snapshot: expected a direct JSON array of 11 source records.' }
foreach ($name in $expectedSources) {
    $expectedPath = [IO.Path]::GetFullPath((Join-Path $installedRoot $name))
    if (@($snapshot | Where-Object { [IO.Path]::GetFullPath($_.path) -eq $expectedPath }).Count -ne 1) { throw "Missing/duplicate source snapshot: $name" }
}
$guardedFiles = @($snapshot) + @(@{path=$checkpoint;sha256=$modelSha}, @{path=(Join-Path $bankPath 'states.npz');sha256=$bankSha})
foreach ($protocol in @('heldout','angle30','angle45')) {
    $r = Get-Content -LiteralPath $reportPaths[$protocol] -Raw | ConvertFrom-Json
    Assert-Report $r $protocol 20
    Assert-HealthyLog (Join-Path $artifactRoot "$RunTag-$protocol.log")
    $guardedFiles += @{path=$reportPaths[$protocol];sha256=(Get-FileHash -LiteralPath $reportPaths[$protocol]).Hash}
}
Assert-HealthyLog (Join-Path $artifactRoot "$RunTag-train.log")
foreach ($name in @('env.yaml','agent.yaml')) {
    $archive = Join-Path $portfolio "configs/$RunTag/$name"
    $sha = (Get-FileHash -LiteralPath (Join-Path $runs[0].FullName "params/$name")).Hash
    if ((Get-FileHash -LiteralPath $archive).Hash -ne $sha) { throw "Archived training config mismatch: $name" }
    $guardedFiles += @{path=$archive;sha256=$sha}
}
function Assert-RecordingReady {
    $active = @(Get-ActiveSimulators)
    if ($active.Count) { throw "Another simulator is active: $($active.ProcessId -join ','). No second simulator may start." }
    foreach ($file in $guardedFiles) {
        if ((Get-FileHash -LiteralPath $file.path -Algorithm SHA256).Hash -ne $file.sha256) { throw "Source/model/report drift: $($file.path)" }
    }
}
Assert-RecordingReady
if ($PreflightOnly) { Write-Output 'READY: all three 20-trial suites, provenance and logs checked. No files created or simulator started.'; return }

New-Item -ItemType Directory -Path $output | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = 'E:\IsaacLab\tmp'; $env:TMP = 'E:\IsaacLab\tmp'
$env:PYTHONIOENCODING = 'utf-8'
Write-Output 'Smith recovery-only diagnostics. Independent matched-condition views, NOT synchronized cameras or a continuous locomotion/recovery controller. Failures remain uncut.'
foreach ($scenario in @('heldout','upright')) {
    foreach ($view in @('front','oblique')) {
        Assert-RecordingReady
        $label = "smith-$scenario-$view"
        $dir = Join-Path $output $label
        $log = Join-Path $output "$label.log"
        if ((Test-Path -LiteralPath $dir) -or (Test-Path -LiteralPath $log)) { throw "Existing stage output: $label" }
        $parameters = @{Checkpoint=$checkpoint;Task=$task;OutputDir=$dir;Trials=1;Seed=20260918;View=$view;Video=$true}
        if ($scenario -eq 'heldout') {
            $parameters.StateBankPath=$bankPath; $parameters.StateBankSplit='heldout'; $parameters.SettleSeconds=1
            $parameters.Poses=@('side','upside_down')
        } else { $parameters.Poses=@('upright'); $parameters.AngleDeg=30 }
        & (Join-Path $PSScriptRoot 'evaluate_recovery.ps1') @parameters *> $log
        if ($LASTEXITCODE -ne 0) { throw "Recording failed; preserve all partial files: $log" }
        Assert-HealthyLog $log
        $recorded = Get-Content -LiteralPath (Join-Path $dir "$($checkpointName)_recovery_metrics.json") -Raw | ConvertFrom-Json
        Assert-Report $recorded $scenario 1 $view
        Write-Output "Recorded $label. Its actual result, not reward or geometry alone, determines success."
    }
    $front = Join-Path $output "smith-$scenario-front/$($checkpointName)_recovery_metrics.json"
    $oblique = Join-Path $output "smith-$scenario-oblique/$($checkpointName)_recovery_metrics.json"
    & $python (Join-Path $PSScriptRoot 'pair_recovery_views.py') $front $oblique (Join-Path $output "smith-$scenario-paired") *> (Join-Path $output "smith-$scenario-pair.log")
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Views NOT paired for ${scenario}. Retain all raw reports/videos and any partial pair outputs; inspect the mismatch. Never force matching outcomes or claim synchronized cameras."
    }
}
Write-Output "Recording finished: $output. Actual visual review of both angles and honest delivery are still REQUIRED. No policy was promoted."
