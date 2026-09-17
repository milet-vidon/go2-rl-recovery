param([switch]$PreflightOnly)
# Resume only the interrupted w30 arm. Never rerun w10 or overwrite old outputs.
$ErrorActionPreference='Stop'
$portfolio=Split-Path $PSScriptRoot -Parent
$artifacts='E:\IsaacLab\artifacts\recovery-20260916'
$runRoot='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery'
$tag='20260916-smithstandresume2900_w30'
$parentRun='2026-09-16_20-41-29_20260916-smithstandpair128x1000_w30'
$parent=Join-Path $runRoot "$parentRun/model_2900.pt"
$bank=Join-Path $portfolio 'datasets/recovery_states/nominal_pd_v1_20260916'
$python='E:\IsaacLab\env\python.exe'
$task='Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0'
$config=Join-Path $portfolio "configs/$tag"
$trainLog=Join-Path $artifacts "$tag-train.log"
$snapshot=Join-Path $artifacts "$tag-source-snapshot.json"

function Assert-Idle {
    $active=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pythonw.exe','kit.exe','isaac-sim.exe') -and
        $_.CommandLine -match 'E:[/\\]IsaacLab' -and
        ($_.Name -in @('kit.exe','isaac-sim.exe') -or $_.CommandLine -match 'rsl_rl[/\\]train.py|resume_smith_checkpoint_entry.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py')
    })
    if($active.Count){throw "Another simulator is active: $($active.ProcessId -join ',')"}
}
function Assert-Healthy([string]$Path) {
    if(Select-String -LiteralPath $Path -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1){throw "Simulator failure: $Path"}
}
$sources=@(Get-Content -LiteralPath (Join-Path $artifacts '20260916-smithstandpair128x1000-source-snapshot.json') -Raw | ConvertFrom-Json)
if($sources.Count -ne 19){throw 'Original source snapshot count changed'}
if((Get-FileHash -LiteralPath $parent).Hash -ne '1F704DA68A0AB1A95E81B9362C7B8F9E47352A354C1ED2CF4EAF0E1EC1D21A45'){throw 'Resume checkpoint mismatch'}
foreach($path in @($parent,(Join-Path $runRoot "$parentRun/params/env.yaml"),(Join-Path $runRoot "$parentRun/params/agent.yaml"),(Join-Path $PSScriptRoot 'resume_smith_w30_2900.ps1'),(Join-Path $PSScriptRoot 'check_smith_resume2900_config.py'))){$sources+=@{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}}
function Assert-Sources { foreach($source in $sources){if((Get-FileHash -LiteralPath $source.path).Hash -ne $source.sha256){throw "Source drift: $($source.path)"}} }
foreach($path in @((Join-Path $PSScriptRoot 'resume_smith_checkpoint_entry.py'),'E:\IsaacLab\repo\scripts\reinforcement_learning\rsl_rl\train.py','E:\IsaacLab\env\Lib\site-packages\rsl_rl\runners\on_policy_runner.py','E:\IsaacLab\env\Lib\site-packages\rsl_rl\algorithms\ppo.py')){$sources+=@{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}}
Assert-Idle; Assert-Sources
if(@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object Name -Like "*_$tag").Count){throw 'Resume run already exists; inspect and resume only missing stages'}
$outputs=@($config,$trainLog,$snapshot)
foreach($protocol in @('heldout','angle30','angle45')){$outputs+=(Join-Path $portfolio "evaluations/$tag-$protocol"),(Join-Path $artifacts "$tag-$protocol.log")}
foreach($path in $outputs){if(Test-Path -LiteralPath $path){throw "Existing output: $path"}}
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONUNBUFFERED='1'; $env:PYTHONIOENCODING='utf-8'
$env:TEMP='E:\IsaacLab\tmp'; $env:TMP='E:\IsaacLab\tmp'
& $python (Join-Path $PSScriptRoot 'check_smith_standweight_config.py') (Join-Path $runRoot "$parentRun/params") --num-envs 128 --iterations 1000 --stand-weight 30
if($LASTEXITCODE -ne 0){throw 'Interrupted run config invalid'}
if($PreflightOnly){'Preflight passed; no simulation or output created.'; return}
$sources | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $snapshot -Encoding UTF8
Push-Location $portfolio
try {
    "$(Get-Date -Format o) Resume w30 after2900, 98 new updates from2901 through2998; restore saved adaptive LR1e-5. Environment/RNG restart, not bitwise uninterrupted."
    $env:OMNI_KIT_ACCEPT_EULA='YES'; $env:OMNI_USER_HOME='E:\IsaacLab\userdata'; $env:OV_USER_HOME='E:\IsaacLab\userdata'
    $env:PIP_CACHE_DIR='E:\IsaacLab\cache\pip'; $env:CONDA_PREFIX='E:\IsaacLab\env'; $env:CONDA_DEFAULT_ENV='isaaclab'
    $env:ISAACLAB_GO2_USD='E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd'
    $env:ISAACLAB_GROUND_USD='E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd'
    $env:ISAACLAB_RECOVERY_BANK_PATH=$bank
    $env:Path='E:\IsaacLab\env;E:\IsaacLab\env\Scripts;'+$env:Path
    foreach($key in @('ISAACLAB_RECOVERY_BANK_COLLECTION','ISAACLAB_RECOVERY_FALL_ANGLE_DEG','ISAACLAB_RECOVERY_CURRICULUM_STEPS','ISAACLAB_RECOVERY_FOCUS_SIDE')){Remove-Item "Env:$key" -ErrorAction SilentlyContinue}
    Push-Location 'E:\IsaacLab\repo'
    try {
        & ./isaaclab.bat -p (Join-Path $PSScriptRoot 'resume_smith_checkpoint_entry.py') --task Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-v0 --num_envs 128 --max_iterations 98 --run_name $tag --device cuda:0 --rendering_mode performance '--kit_args=--/app/vulkan=false' --resume --load_run $parentRun --checkpoint model_2900.pt --headless env.rewards.smith_stand.weight=30.0 *> $trainLog
    } finally {Pop-Location}
    if($LASTEXITCODE -ne 0){throw 'Resume training failed'}
    Assert-Healthy $trainLog; Assert-Sources; Assert-Idle
    if(-not(Select-String -LiteralPath $trainLog -SimpleMatch 'EXACT_RESUME_STATE')){throw 'Missing exact resume state confirmation'}
    $runs=@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object Name -Like "*_$tag")
    if($runs.Count -ne 1){throw 'Expected exactly one resumed run'}
    $checkpoint=Join-Path $runs[0].FullName 'model_2998.pt'
    if(-not(Test-Path -LiteralPath $checkpoint)){throw 'Final2998 missing'}
    & $python ./scripts/check_smith_resume2900_config.py (Join-Path $runs[0].FullName 'params')
    if($LASTEXITCODE -ne 0){throw 'Resumed configuration drift'}
    New-Item -ItemType Directory -Path $config | Out-Null
    foreach($name in @('env.yaml','agent.yaml')){Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$name") -Destination (Join-Path $config $name)}
    "$(Get-Date -Format o) Completed remaining training: $checkpoint SHA256 $((Get-FileHash -LiteralPath $checkpoint).Hash)"
    foreach($protocol in @('heldout','angle30','angle45')){
        Assert-Idle; Assert-Sources
        $directory=Join-Path $portfolio "evaluations/$tag-$protocol"
        $evalParams=@{Checkpoint=$checkpoint;Task=$task;OutputDir=$directory;Trials=20}
        if($protocol -eq 'heldout'){$evalParams.StateBankPath=$bank;$evalParams.StateBankSplit='heldout';$evalParams.SettleSeconds=1;$evalParams.Seed=20260918;$evalParams.Poses=@('side','upside_down')}
        else{$evalParams.AngleDeg=if($protocol -eq 'angle30'){30}else{45};$evalParams.Seed=if($protocol -eq 'angle30'){20260918}else{20260916};$evalParams.Poses=if($protocol -eq 'angle30'){@('upright','side','fore_aft')}else{@('upright','side','fore_aft','upside_down')}}
        $log=Join-Path $artifacts "$tag-$protocol.log"
        & ./scripts/evaluate_recovery.ps1 @evalParams *> $log
        if($LASTEXITCODE -ne 0){throw "Evaluation failed: $protocol"}
        Assert-Healthy $log
        $report=Join-Path $directory 'model_2998.pt_recovery_metrics.json'
        $result=Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
        if($result.task -ne $task -or $result.seed -ne $evalParams.Seed -or [IO.Path]::GetFullPath($result.checkpoint) -ne [IO.Path]::GetFullPath($checkpoint) -or $result.checkpoint_sha256 -ne (Get-FileHash -LiteralPath $checkpoint).Hash -or $result.self_collisions_enabled -ne $true -or $null -ne $result.video_view -or $result.action_representation.reference -ne 'nominal' -or $result.action_representation.scale -ne 0.25 -or $result.action_representation.sample_period_s -ne 0.02 -or $result.action_representation.held_over_physics_substeps -ne 4){throw 'Evaluation identity mismatch'}
        if((@($result.results.PSObject.Properties.Name | Sort-Object) -join ',') -ne (@($evalParams.Poses | Sort-Object) -join ',')){throw 'Pose set mismatch'}
        foreach($r in $result.results.PSObject.Properties.Value){if($r.trials -ne 20 -or $r.stable_hold_s -ne 3 -or $r.horizon_s -ne 8){throw 'Acceptance protocol changed'}}
        if($protocol -eq 'heldout'){if($result.state_bank.sha256 -ne (Get-FileHash -LiteralPath (Join-Path $bank 'states.npz')).Hash -or $result.state_bank.split -ne 'heldout' -or $result.settle_actual_s -ne 1 -or $result.settle_control_steps -ne 50 -or $null -ne $result.angle_deg){throw 'Bank protocol mismatch'}}
        elseif($result.angle_deg -ne $evalParams.AngleDeg -or $result.settle_actual_s -ne 0 -or $null -ne $result.state_bank){throw 'Controlled-drop protocol mismatch'}
        & $python ./scripts/audit_recovery_report.py $report
        if($LASTEXITCODE -ne 0){throw 'Report inconsistency'}
        "$(Get-Date -Format o) Completed $protocol. Consistency is not a performance pass."
    }
    'Finite resumed w30 training and all three evaluations completed. No promotion or automatic extension.'
} finally {Pop-Location}
