param([string]$IsaacLabRepo = 'E:\IsaacLab\repo', [switch]$DryRun)
$ErrorActionPreference = 'Stop'
$portfolio = Split-Path $PSScriptRoot -Parent
$destination = Join-Path $IsaacLabRepo 'source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2'
if (-not (Test-Path (Join-Path $destination 'flat_env_cfg.py'))) { throw 'Provide an installed Isaac Lab repository with the Go2 flat task.' }
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($IsaacLabRepo)) -eq 'C:\') { throw 'This project stores generated code and backups outside C:.' }
$backup = Join-Path (Split-Path $IsaacLabRepo -Parent) ('artifacts\install-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$copies = @()
foreach ($name in @('standard_env_cfg.py', 'robust_env_cfg.py', 'natural_env_cfg.py', 'natural_stop_env_cfg.py', 'recovery_env_cfg.py', 'recovery_mdp.py', 'recovery_math.py')) {
    $copies += @{ Source = Join-Path $portfolio "src\go2_recovery\$name"; Target = Join-Path $destination $name }
}
$copies += @{ Source = Join-Path $portfolio 'src\go2_recovery\go2_registry_snapshot.py'; Target = Join-Path $destination '__init__.py' }
$copies += @{ Source = Join-Path $portfolio 'src\go2_recovery\go2_ppo_snapshot.py'; Target = Join-Path $destination 'agents\rsl_rl_ppo_cfg.py' }
$copies += @{ Source = Join-Path $portfolio 'scripts\evaluate_go2_stand_walk_stop.py'; Target = Join-Path $IsaacLabRepo 'scripts\environments\evaluate_go2_stand_walk_stop.py' }
foreach ($item in $copies) {
    if (-not (Test-Path -LiteralPath $item.Source)) { throw "Missing source: $($item.Source)" }
    Write-Host "$($item.Source) -> $($item.Target)"
}
$patches = @(
    @{ Patch = 'local-go2.patch'; File = 'source\isaaclab_assets\isaaclab_assets\robots\unitree.py' },
    @{ Patch = 'local-ground.patch'; File = 'source\isaaclab\isaaclab\sim\spawners\from_files\from_files_cfg.py' }
)
$pendingPatches = @()
foreach ($item in $patches) {
    $item.Path = Join-Path $portfolio ('patches\' + $item.Patch)
    & git -C $IsaacLabRepo apply --reverse --check $item.Path 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "$($item.Patch): already installed"; continue }
    & git -C $IsaacLabRepo apply --check $item.Path
    if ($LASTEXITCODE -ne 0) { throw "Patch conflicts with the installed Isaac Lab: $($item.Patch). No files were copied." }
    $pendingPatches += $item
}
if ($DryRun) { return }
New-Item -ItemType Directory -Path $backup -Force | Out-Null
for ($i = 0; $i -lt $copies.Count; $i++) {
    $item = $copies[$i]
    if (Test-Path -LiteralPath $item.Target) {
        Copy-Item -LiteralPath $item.Target -Destination (Join-Path $backup "$i-$(Split-Path $item.Target -Leaf)")
    }
    Copy-Item -LiteralPath $item.Source -Destination $item.Target -Force
}
foreach ($item in $pendingPatches) {
    Copy-Item -LiteralPath (Join-Path $IsaacLabRepo $item.File) -Destination (Join-Path $backup ($item.Patch + '.original'))
    & git -C $IsaacLabRepo apply $item.Path
    if ($LASTEXITCODE -ne 0) { throw "Patch failed: $($item.Patch); backups: $backup" }
}
Write-Host "Installed task overlay. Previous files are preserved in $backup"
