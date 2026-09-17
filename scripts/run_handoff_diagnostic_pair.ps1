param([switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$portfolio=Split-Path $PSScriptRoot -Parent
$artifacts='E:\IsaacLab\artifacts\recovery-20260916'
$roll='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-16_22-52-12_20260916-smithstandresume2900_w30\model_2998.pt'
$stand=Join-Path $portfolio 'models/recovery/recovery_aligned_model3448.pt'
$bank=Join-Path $portfolio 'datasets/recovery_states/nominal_pd_v1_20260916'
$old=Join-Path $portfolio 'evaluations/20260916-smithstandresume2900_w30-heldout'
$python='E:\IsaacLab\env\python.exe'
function Assert-Idle {
    $active=@(Get-CimInstance Win32_Process|Where-Object{
        $_.Name -in @('python.exe','pythonw.exe','kit.exe','isaac-sim.exe') -and $_.CommandLine -match 'E:[/\\]IsaacLab' -and
        ($_.Name -in @('kit.exe','isaac-sim.exe') -or $_.CommandLine -match 'rsl_rl[/\\]train.py|resume_smith_checkpoint_entry.py|evaluate_go2_|build_recovery_state_bank.py')
    })
    if($active.Count){throw "Simulator active: $($active.ProcessId -join ',')"}
}
Assert-Idle
if((Get-FileHash -LiteralPath $roll).Hash -ne '65A9278EF212FD27FA47622E4506A1DB6F8BBECC9FA5F15E1F6CE0E2808CD464'){throw 'Roll model mismatch'}
if((Get-FileHash -LiteralPath $stand).Hash -ne '1F546523BAA57C7997AD2883689667D6E738EAC098F14FB5FA595AAE778A2DE2'){throw 'Stand model mismatch'}
foreach($mode in @('control','dual')){
    foreach($path in @((Join-Path $portfolio "evaluations/20260916-handoff2998-$mode-heldout"),(Join-Path $artifacts "handoff2998-$mode-heldout.log"))){
        if(Test-Path -LiteralPath $path){throw "Existing output: $path"}
    }
}
$sources=@()
foreach($path in @((Join-Path $PSScriptRoot 'evaluate_go2_recovery.py'),(Join-Path $PSScriptRoot 'evaluate_recovery.ps1'),(Join-Path $portfolio 'src/go2_recovery/recovery_handoff_math.py'),$roll,$stand,(Join-Path $bank 'states.npz'))){$sources+=@{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}}
function Assert-Sources {foreach($s in $sources){if((Get-FileHash -LiteralPath $s.path).Hash -ne $s.sha256){throw "Runtime drift: $($s.path)"}}}
if($PreflightOnly){'Preflight passed; no simulator or output created.';return}
$env:PYTHONDONTWRITEBYTECODE='1';$env:TEMP='E:\IsaacLab\tmp';$env:TMP='E:\IsaacLab\tmp'
Push-Location $portfolio
try {
    foreach($mode in @('control','dual')){
        Assert-Idle;Assert-Sources
        $out=Join-Path $portfolio "evaluations/20260916-handoff2998-$mode-heldout"
        $log=Join-Path $artifacts "handoff2998-$mode-heldout.log"
        $params=@{Checkpoint=$roll;Task='Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0';OutputDir=$out;Trials=20;Seed=20260918;StateBankPath=$bank;StateBankSplit='heldout';SettleSeconds=1;Poses=@('side','upside_down')}
        if($mode -eq 'dual'){$params.StandCheckpoint=$stand}
        "$(Get-Date -Format o) Starting $mode,20trials per pose; same frozen models and start protocol."
        & ./scripts/evaluate_recovery.ps1 @params *> $log
        if($LASTEXITCODE -ne 0){throw "Failed $mode"}
        if(Select-String -LiteralPath $log -Pattern 'Traceback \(most recent|CUDA.*out of memory|CUDA error|PhysX error|discard.*contacts|buffer.*overflow'|Select-Object -First 1){throw "Simulation error: $log"}
        Assert-Sources;Assert-Idle
        $report=Join-Path $out 'model_2998.pt_recovery_metrics.json'
        if($mode -eq 'control'){
            & $python ./scripts/audit_recovery_report.py $report
            if($LASTEXITCODE -ne 0){throw 'Control report inconsistent'}
            $a=Get-Content -LiteralPath (Join-Path $old 'model_2998.pt_recovery_metrics.json') -Raw|ConvertFrom-Json
            $b=Get-Content -LiteralPath $report -Raw|ConvertFrom-Json
            if(($a.results|ConvertTo-Json -Depth 40 -Compress) -ne ($b.results|ConvertTo-Json -Depth 40 -Compress)){throw 'Default single-policy results regression; preserve outputs and investigate'}
            if((Get-FileHash -LiteralPath (Join-Path $old 'model_2998.pt_recovery_trace.csv')).Hash -ne (Get-FileHash -LiteralPath (Join-Path $out 'model_2998.pt_recovery_trace.csv')).Hash){throw 'Default single-policy trace differs; preserve and investigate'}
            'Single-policy results and trace are EXACTLY equal to the archived w30 control.'
        } else {
            $a=Get-Content -LiteralPath (Join-Path $portfolio 'evaluations/20260916-handoff2998-control-heldout/model_2998.pt_recovery_metrics.json') -Raw|ConvertFrom-Json
            $b=Get-Content -LiteralPath $report -Raw|ConvertFrom-Json
            foreach($pose in @('side','upside_down')){
                foreach($field in @('release_state_before_settling','policy_start_state','state_bank_selection')){
                    if(($a.results.$pose.$field|ConvertTo-Json -Depth 40 -Compress) -ne ($b.results.$pose.$field|ConvertTo-Json -Depth 40 -Compress)){throw "Start mismatch: $pose/$field"}
                }
            }
            'Dual-policy starts and bank IDs match the control exactly. Independently audit the NEW dual schema before claiming outcomes.'
        }
        "$(Get-Date -Format o) Completed $mode. Not a promotion or visual acceptance."
    }
} finally {Pop-Location}
