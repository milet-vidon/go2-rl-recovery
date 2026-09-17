param(
    [Parameter(Mandatory=$true)][ValidatePattern('^20260917-commonphysics[A-Za-z0-9_-]+$')][string]$RunTag,
    [ValidateSet('smoke','formal')][string]$Mode = 'smoke',
    [string]$SmokeReceipt,
    [switch]$PreflightOnly
)
# One explicit finite run only. Formal requires a same-source completed smoke;
# it still resumes ORIGINAL3947, never the smoke or a rejected speed checkpoint.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$taskPortfolio='E:\IsaacLab\go2-rl-open-source'
$taskRepo='E:\IsaacLab\repo'
$taskPython='E:\IsaacLab\env\python.exe'
$taskEntry=Join-Path $PSScriptRoot 'train_common_physics_adaptation.py'
$taskParent='2026-09-17_03-10-21_20260917-speedretention128x300'
$taskParentSha='3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f'
$taskNum=if($Mode -eq 'smoke'){16}else{128}
$taskUpdates=if($Mode -eq 'smoke'){2}else{100}
$taskArtifacts=Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
$taskConfigs=Join-Path $taskPortfolio "configs/$RunTag"
$taskRunRoot=Join-Path $taskRepo 'logs/rsl_rl/unitree_go2_flat'
$taskExpectedIteration=3947+$taskUpdates-1
$taskExpectedAdam=79120+20*$taskUpdates
$taskExpectedSteps=$taskNum*24*$taskUpdates
function Assert-CommonPhysicsIdle {
    $taskActive=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit.*|isaac-sim.*)\.exe$' -and
        ($_.ExecutablePath -match '^E:[/\\]IsaacLab' -or $_.CommandLine -match 'E:[/\\]IsaacLab')})
    if($taskActive.Count){throw "E-workspace Python/Kit active: $($taskActive.ProcessId -join ','). No GPU overlap or automatic termination."}
}
Assert-CommonPhysicsIdle
foreach($taskPath in @($taskArtifacts,$taskConfigs)){
    if([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($taskPath)) -ne 'E:\'){throw 'All outputs must remain on E:'}
    if(Test-Path -LiteralPath $taskPath){throw "Preserve existing output: $taskPath"}
}
if(@(Get-ChildItem -LiteralPath $taskRunRoot -Directory | Where-Object {$_.Name.EndsWith("_$RunTag")}).Count){throw 'Run tag exists'}
if($Mode -eq 'formal'){
    if(-not $SmokeReceipt -or -not (Test-Path -LiteralPath $SmokeReceipt -PathType Leaf)){throw 'Formal requires the successful same-source smoke result path.'}
    $SmokeReceipt=(Resolve-Path -LiteralPath $SmokeReceipt).Path
    if([IO.Path]::GetPathRoot($SmokeReceipt) -ne 'E:\'){throw 'Smoke receipt must remain on E:'}
} elseif($SmokeReceipt){throw 'Smoke independently resumes original3947 and accepts no prior candidate.'}
$env:OMNI_KIT_ACCEPT_EULA='YES'
$env:OMNI_USER_HOME='E:\IsaacLab\userdata'
$env:OV_USER_HOME='E:\IsaacLab\userdata'
$env:TEMP='E:\IsaacLab\tmp'
$env:TMP='E:\IsaacLab\tmp'
$env:PIP_CACHE_DIR='E:\IsaacLab\cache\pip'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:ISAACLAB_GO2_USD='E:\IsaacLab\userdata\assets\Robots\Unitree\Go2\go2.usd'
$env:ISAACLAB_GROUND_USD='E:\IsaacLab\userdata\assets\Environments\Grid\default_environment.usd'
$env:CONDA_PREFIX='E:\IsaacLab\env'
$env:Path='E:\IsaacLab\env;E:\IsaacLab\env\Scripts;'+$env:Path
$taskArgs=@('--task','Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0','--num_envs',"$taskNum",
    '--max_iterations',"$taskUpdates",'--run_name',$RunTag,'--device','cuda:0',
    '--rendering_mode','performance','--kit_args=--/app/vulkan=false','--headless',
    '--resume','--load_run',$taskParent,'--checkpoint','model_3947.pt','agent.algorithm.learning_rate=1e-5')
if($Mode -eq 'formal'){$taskArgs+=@('--smoke_receipt',$SmokeReceipt)}
Push-Location $taskRepo
try {
    Assert-CommonPhysicsIdle
    $taskPreflight=& $taskPython -B $taskEntry @taskArgs --preflight-only 2>&1
    if($LASTEXITCODE -ne 0){throw "Preflight failed: $($taskPreflight -join [Environment]::NewLine)"}
    $taskPreflightText=$taskPreflight -join [Environment]::NewLine
    $taskBefore=$taskPreflightText | ConvertFrom-Json
    if($taskBefore.preflight_pass -ne $true -or $taskBefore.simulator_started -ne $false -or
       $taskBefore.parent.sha256 -ne $taskParentSha -or $taskBefore.expected_adam_step -ne $taskExpectedAdam){throw 'Invalid CPU preflight receipt'}
    if($PreflightOnly){Write-Output $taskPreflightText; return}
    $null=New-Item -ItemType Directory -Path $taskArtifacts
    [IO.File]::WriteAllText((Join-Path $taskArtifacts 'preflight.json'),$taskPreflightText,[Text.UTF8Encoding]::new($false))
    $taskInvocation=[ordered]@{protocol='bounded_common_physics_launcher_v1';status='running';mode=$Mode;
        run_tag=$RunTag;started_at=(Get-Date).ToString('o');args=$taskArgs;smoke_receipt=$SmokeReceipt;
        expected_iteration=$taskExpectedIteration;expected_adam_step=$taskExpectedAdam;
        expected_environment_steps=$taskExpectedSteps;quality_accepted=$false;promotion_performed=$false}
    $taskInvocationPath=Join-Path $taskArtifacts 'launcher_invocation.json'
    function Save-CommonPhysicsInvocation {
        [IO.File]::WriteAllText($taskInvocationPath,($taskInvocation|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    }
    Save-CommonPhysicsInvocation
    try {
        Assert-CommonPhysicsIdle
        $taskLog=Join-Path $taskArtifacts 'train.log'
        & ./isaaclab.bat -p $taskEntry @taskArgs *> $taskLog
        $taskCode=$LASTEXITCODE
        if($taskCode -ne 0){throw "Finite training failed exit=$taskCode; preserve $taskLog"}
        $taskBad=@(rg -n -i 'Traceback \(most recent|CUDA error|CUDA_ERROR|CUDA.*out of memory|\[Error\].*PhysX|PhysX.*overflow|buffer.*overflow|discard.*contacts|contacts.*discard|Fatal Python error|illegal memory access' $taskLog)
        if($LASTEXITCODE -notin @(0,1)){throw 'Could not inspect training log'}
        if($taskBad.Count){throw "Invalid simulator log: $($taskBad -join ' | ')"}
        Assert-CommonPhysicsIdle
        $taskRuns=@(Get-ChildItem -LiteralPath $taskRunRoot -Directory | Where-Object {$_.Name.EndsWith("_$RunTag")})
        if($taskRuns.Count -ne 1){throw 'Expected exactly one new run'}
        $taskResultPath=Join-Path $taskRuns[0].FullName 'common_physics_training_result.json'
        $taskResult=Get-Content -LiteralPath $taskResultPath -Raw | ConvertFrom-Json
        if($taskResult.protocol -ne 'bounded_common_physics_adaptation_training_v1' -or
           $taskResult.completion_verified -ne $true -or $taskResult.run_name -ne $RunTag -or
           $taskResult.parent.sha256 -ne $taskParentSha -or $taskResult.num_envs -ne $taskNum -or
           $taskResult.updates -ne $taskUpdates -or $taskResult.actual_environment_steps -ne $taskExpectedSteps -or
           $taskResult.actual_control_steps -ne (24*$taskUpdates) -or $taskResult.quality_accepted -ne $false -or
           $taskResult.promotion_performed -ne $false -or $taskResult.hardware_execution -ne $false -or
           $taskResult.checkpoint_metadata.iter -ne $taskExpectedIteration -or
           $taskResult.checkpoint_metadata.tensor_count -ne 68 -or $taskResult.checkpoint_metadata.all_tensors_finite -ne $true -or
           @($taskResult.checkpoint_metadata.adam_steps).Count -ne 17 -or
           @($taskResult.checkpoint_metadata.adam_steps | Where-Object {$_ -ne $taskExpectedAdam}).Count){throw 'Invalid finite training result'}
        $taskExpectedCheckpoint=Join-Path $taskRuns[0].FullName "model_$taskExpectedIteration.pt"
        if([IO.Path]::GetFullPath($taskResult.checkpoint) -ne $taskExpectedCheckpoint -or
           (Get-FileHash -LiteralPath $taskExpectedCheckpoint -Algorithm SHA256).Hash -ne $taskResult.checkpoint_metadata.sha256){throw 'Final checkpoint identity mismatch'}
        foreach($taskSource in $taskResult.source_snapshot){
            if((Get-FileHash -LiteralPath $taskSource.path -Algorithm SHA256).Hash -ne $taskSource.sha256){throw "Preserved source/model changed: $($taskSource.path)"}
        }
        foreach($taskYaml in $taskResult.actual_yaml){
            if((Get-FileHash -LiteralPath $taskYaml.path -Algorithm SHA256).Hash -ne $taskYaml.sha256){throw 'Actual YAML changed'}
        }
        & $taskPython -I -B (Join-Path $PSScriptRoot 'verify_overnight_baselines.py')
        if($LASTEXITCODE -ne 0){throw '16-model byte preservation failed'}
        $null=New-Item -ItemType Directory -Path $taskConfigs
        foreach($taskName in @('env.yaml','agent.yaml')){
            Copy-Item -LiteralPath (Join-Path $taskRuns[0].FullName "params/$taskName") -Destination (Join-Path $taskConfigs $taskName)
        }
        $taskInvocation.status='completed_training_only'
        $taskInvocation['training_result']=$taskResultPath
        $taskInvocation['training_result_sha256']=(Get-FileHash -LiteralPath $taskResultPath -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Output "FINITE_COMMON_PHYSICS_COMPLETE: $taskExpectedCheckpoint"
        Write-Output 'Training only. No further run, promotion, fast-running or integration claim. Evaluate both recovery and original physics.'
    } catch {
        $taskInvocation.status='error'
        $taskInvocation['error']=$_.Exception.Message
        throw
    } finally {
        $taskInvocation['finished_at']=(Get-Date).ToString('o')
        Save-CommonPhysicsInvocation
    }
} finally {Pop-Location}
