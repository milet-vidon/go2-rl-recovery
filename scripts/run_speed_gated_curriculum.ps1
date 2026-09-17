param(
    [Parameter(Mandatory=$true)][ValidatePattern('^20260917-speedgate[A-Za-z0-9_-]+$')][string]$RunTag,
    [ValidateSet('smoke','formal')][string]$Mode = 'smoke'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$portfolio=Split-Path $PSScriptRoot -Parent
$repo='E:\IsaacLab\repo'
$python='E:\IsaacLab\env\python.exe'
$parent='2026-09-17_03-10-21_20260917-speedretention128x300'
$num=if($Mode -eq 'smoke'){16}else{128}
$updates=if($Mode -eq 'smoke'){2}else{50}
$artifacts=Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
$configs=Join-Path $portfolio "configs/$RunTag"
$runRoot=Join-Path $repo 'logs/rsl_rl/unitree_go2_flat'
function Assert-Idle {
    $active=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit|isaac-sim.*)\.exe$' -and
        ($_.ExecutablePath -match '^E:[/\\]IsaacLab' -or $_.CommandLine -match 'E:[/\\]IsaacLab')})
    if($active.Count){throw "Workspace Python/Kit active: $($active.ProcessId -join ',')"}
}
Assert-Idle
foreach($path in @($artifacts,$configs)){
    if(Test-Path -LiteralPath $path){throw "Preserve existing output: $path"}
}
if(@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object {$_.Name.EndsWith("_$RunTag")}).Count){throw 'Run tag exists'}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUNBUFFERED='1'
Push-Location $portfolio
try {
    & ./scripts/train.ps1 -Stage natural_robust_push -NumEnvs $num -MaxIterations $updates -LoadRun $parent -Checkpoint model_3947.pt -RunName $RunTag -Headless -DryRun
    & $python -I -B ./scripts/verify_overnight_baselines.py
    if($LASTEXITCODE -ne 0){throw 'Preservation failed'}
    New-Item -ItemType Directory -Path $artifacts | Out-Null
    $argsTrain=@('-p',(Join-Path $PSScriptRoot 'train_speed_gated_curriculum.py'),
        '--task','Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0','--num_envs',"$num",
        '--max_iterations',"$updates",'--run_name',$RunTag,'--device','cuda:0',
        '--rendering_mode','performance','--kit_args=--/app/vulkan=false','--headless',
        '--resume','--load_run',$parent,'--checkpoint','model_3947.pt','agent.algorithm.learning_rate=1e-5')
    Assert-Idle
    Push-Location $repo
    try { & ./isaaclab.bat @argsTrain *> (Join-Path $artifacts 'train.log'); $code=$LASTEXITCODE }
    finally { Pop-Location }
    & $python -I -B ./scripts/verify_overnight_baselines.py
    if($LASTEXITCODE -ne 0){throw 'Post-training preservation failed'}
    if($code -ne 0){throw "Training failed: $artifacts/train.log"}
    $bad=Select-String -LiteralPath (Join-Path $artifacts 'train.log') -Pattern 'Traceback \(most recent|CUDA error|CUDA.*out of memory|\[Error\].*PhysX|buffer.*overflow|discard.*contacts' | Select-Object -First 1
    if($null -ne $bad){throw "Invalid simulator log: $($bad.Line)"}
    $runs=@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object {$_.Name.EndsWith("_$RunTag")})
    if($runs.Count -ne 1){throw 'Expected one new run'}
    $receipt=Join-Path $runs[0].FullName 'speed_training_result.json'
    $r=Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
    if($r.status -ne 'finite_training_complete' -or $r.updates -ne $updates -or $r.num_envs -ne $num -or $r.quality_accepted -ne $false){throw 'Invalid training completion receipt'}
    New-Item -ItemType Directory -Path $configs | Out-Null
    foreach($name in @('env.yaml','agent.yaml')){
        Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$name") -Destination (Join-Path $configs $name)
    }
    Write-Output "FINITE_SPEED_GATE_COMPLETE: $($r.checkpoint)"
    Write-Output 'Not promoted; full18 behavioral screen required. No automatic further training block.'
} finally { Pop-Location }
