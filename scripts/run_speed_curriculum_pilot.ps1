param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag = '20260917-speedcurriculum128x300',
    [ValidateSet(2,300)][int]$Iterations = 300,
    [ValidateSet(16,128)][int]$NumEnvs = 128,
    [switch]$PreflightOnly
)
# TRAINING ONLY finite one-variable pilot; evaluation matrix is a separate stage.
# No promotion, acceptance claim, endless continuation, or modifications to train.ps1.
$ErrorActionPreference = 'Stop'
if (-not (($NumEnvs -eq 16 -and $Iterations -eq 2) -or ($NumEnvs -eq 128 -and $Iterations -eq 300))) {
    throw 'Use 16 envs/2 updates for smoke or 128 envs/300 updates for the formal pilot.'
}
$portfolio = Split-Path $PSScriptRoot -Parent
$repo = 'E:\IsaacLab\repo'
$python = 'E:\IsaacLab\env\python.exe'
$runRoot = Join-Path $repo 'logs/rsl_rl/unitree_go2_flat'
$installed = Join-Path $repo 'source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2'
$parentRun = '2026-09-09_15-50-37_natural_robust_push_20260909'
$parent = Join-Path $runRoot "$parentRun/model_3648.pt"
$parentSha = '475F07BF0B05101DD4157212B4E3FD48A04731D452D7A349FE467C508DBAAAC8'
$logRoot = Join-Path 'E:\IsaacLab\artifacts\recovery-20260917' $RunTag
$config = Join-Path $portfolio "configs/$RunTag"
$snapshot = Join-Path $logRoot 'source-snapshot.json'
$trainLog = Join-Path $logRoot 'train.log'
$checkLog = Join-Path $logRoot 'config-checkpoint-check.log'
$expectedFinal = "model_$(3648 + $Iterations - 1).pt"

function Assert-Idle {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pythonw.exe','kit.exe','isaac-sim.exe') -and
        $_.CommandLine -match 'E:[/\\]IsaacLab' -and
        ($_.Name -in @('kit.exe','isaac-sim.exe') -or $_.CommandLine -match 'rsl_rl[/\\]train.py|resume_smith_checkpoint_entry.py|evaluate_go2_|build_recovery_state_bank.py|validate_recovery_bank_live.py')
    })
    if ($active.Count) { throw "Another simulator is active: $($active.ProcessId -join ',')." }
}
function Assert-HealthyLog([string]$Path) {
    $bad = Select-String -LiteralPath $Path -Pattern 'CUDA error|CUDA.*out of memory|PhysX error|\[Error\].*PhysX|Failed to get DOF|Traceback \(most recent|buffer.*overflow|discard.*contacts|contacts.*discard' | Select-Object -First 1
    if ($null -ne $bad) { throw "Invalid simulation log: $Path : $($bad.Line)" }
}
Assert-Idle
foreach ($path in @($portfolio,$repo,$python,$runRoot,$installed,$parent,$logRoot,$config)) {
    if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($path)) -ne 'E:\') { throw "Non-E-drive path: $path" }
}
foreach ($path in @($logRoot,$config)) {
    if (Test-Path -LiteralPath $path) { throw "Existing output: $path. Preserve partial evidence; never rerun over it." }
}
if (@(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") }).Count) { throw 'Existing training run tag.' }
if ((Get-FileHash -LiteralPath $parent -Algorithm SHA256).Hash -ne $parentSha) { throw 'Parent checkpoint SHA256 mismatch.' }
$sources = @(@{path=$parent;sha256=$parentSha})
foreach ($name in @('env.yaml','agent.yaml')) {
    $actual = Join-Path $runRoot "$parentRun/params/$name"
    $archive = Join-Path $portfolio "configs/natural-robust-push-20260909/$name"
    if ((Get-FileHash -LiteralPath $actual).Hash -ne (Get-FileHash -LiteralPath $archive).Hash) { throw "Parent archived config mismatch: $name" }
    $sources += @{path=$actual;sha256=(Get-FileHash -LiteralPath $actual).Hash}, @{path=$archive;sha256=(Get-FileHash -LiteralPath $archive).Hash}
}
foreach ($name in @('natural_env_cfg.py','natural_stop_env_cfg.py','natural_robust_env_cfg.py','__init__.py','agents/rsl_rl_ppo_cfg.py')) {
    $sourceName = switch ($name) { '__init__.py' {'go2_registry_snapshot.py'} 'agents/rsl_rl_ppo_cfg.py' {'go2_ppo_snapshot.py'} default {$name} }
    $actual = Join-Path $installed $name
    $source = Join-Path $portfolio "src/go2_recovery/$sourceName"
    if ((Get-FileHash -LiteralPath $actual).Hash -ne (Get-FileHash -LiteralPath $source).Hash) { throw "Installed/source overlay mismatch: $name" }
    $sources += @{path=$source;sha256=(Get-FileHash -LiteralPath $source).Hash}
}
# Hash all installed Go2 Python modules, including inherited task/reward code.
foreach ($item in Get-ChildItem -LiteralPath $installed -Recurse -File -Filter '*.py') {
    $sources += @{path=$item.FullName;sha256=(Get-FileHash -LiteralPath $item.FullName).Hash}
}
foreach ($path in @((Join-Path $PSScriptRoot 'train.ps1'),$PSCommandPath,
                    (Join-Path $PSScriptRoot 'check_speed_curriculum_config.py'),
                    (Join-Path $PSScriptRoot 'check_smith_standweight_config.py'),
                    (Join-Path $repo 'scripts/reinforcement_learning/rsl_rl/train.py'),
                    'E:\IsaacLab\env\Lib\site-packages\rsl_rl\runners\on_policy_runner.py',
                    'E:\IsaacLab\env\Lib\site-packages\rsl_rl\algorithms\ppo.py')) {
    $sources += @{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}
}
function Assert-Sources {
    foreach ($source in $sources) {
        if ((Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash -ne $source.sha256) { throw "Source/parent drift: $($source.path)" }
    }
}
Write-Output "Finite SPEED TRAINING ONLY: parent3648, $NumEnvs envs/$Iterations updates, $($NumEnvs * $Iterations * 24) environment steps, expected $expectedFinal."
Write-Output 'Command change: lin_vel_x [-0.5,1.0] -> [-0.5,1.2]. Match adaptive LR scalar to saved 1e-5; not exact RNG/simulator continuation.'
if ($PreflightOnly) { Write-Output 'Preflight passed; no simulator started and no output created.'; return }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TEMP = 'E:\IsaacLab\tmp'
$env:TMP = 'E:\IsaacLab\tmp'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Path $logRoot | Out-Null
$sources | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $snapshot -Encoding UTF8
Push-Location $portfolio
try {
    Assert-Idle
    Assert-Sources
    # Reuse the existing wrapper only for its E-drive runtime setup. Its DryRun
    # returns before launching. The command-range override is the experimental
    # change; a separate explicit LR override matches the parent's saved optimizer.
    & ./scripts/train.ps1 -Stage natural_robust_push -NumEnvs $NumEnvs -MaxIterations $Iterations -LoadRun $parentRun -Checkpoint model_3648.pt -RunName $RunTag -Headless -DryRun
    $argsTrain = @('-p','scripts\reinforcement_learning\rsl_rl\train.py',
        '--task','Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0','--num_envs',"$NumEnvs",
        '--max_iterations',"$Iterations",'--run_name',$RunTag,'--device','cuda:0',
        '--rendering_mode','performance','--kit_args=--/app/vulkan=false','--headless',
        '--resume','--load_run',$parentRun,'--checkpoint','model_3648.pt',
        'env.commands.base_velocity.ranges.lin_vel_x=[-0.5,1.2]',
        'agent.algorithm.learning_rate=1e-5')
    Push-Location $repo
    try { & .\isaaclab.bat @argsTrain *> $trainLog }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $trainLog" }
    Assert-HealthyLog $trainLog
    Assert-Sources
    $runs = @(Get-ChildItem -LiteralPath $runRoot -Directory | Where-Object { $_.Name.EndsWith("_$RunTag") })
    if ($runs.Count -ne 1) { throw 'Expected exactly one speed pilot run.' }
    $checkpoint = Join-Path $runs[0].FullName $expectedFinal
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Missing final checkpoint: $checkpoint" }
    New-Item -ItemType Directory -Path $config | Out-Null
    foreach ($name in @('env.yaml','agent.yaml')) { Copy-Item -LiteralPath (Join-Path $runs[0].FullName "params/$name") -Destination (Join-Path $config $name) }
    & $python ./scripts/check_speed_curriculum_config.py $config --num-envs $NumEnvs --iterations $Iterations --run-name $RunTag --checkpoint $checkpoint *> $checkLog
    if ($LASTEXITCODE -ne 0) { throw "Config/checkpoint guard failed: $checkLog" }
    Assert-Sources
    Get-Content -LiteralPath $checkLog
    Write-Output "Completed finite TRAINING ONLY: $checkpoint; SHA256 $((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash)."
    Write-Output 'No promotion or evaluation acceptance. Next: separate fresh-scene 3-seed x 0.5/0.8/1.0 m/s matrix and separate drift screening, including failed cases.'
}
finally { Pop-Location }
