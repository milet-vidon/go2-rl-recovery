param([ValidatePattern('^20260917-handoff-combined-[A-Za-z0-9_-]+$')][string]$RunTag='20260917-handoff-combined-v1',
      [switch]$ReuseV1OffUpright)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path $PSScriptRoot -Parent
$out=Join-Path $root "evaluations/$RunTag"
$python='E:\IsaacLab\env\python.exe'
$roll='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_recovery\2026-09-16_19-02-40_20260916-smithnominal128x2000\model_1999.pt'
$stand='E:\IsaacLab\repo\logs\rsl_rl\unitree_go2_handoff_stand\20260917-handoff-stand128x100\model_3547.pt'
$wrapper=Join-Path $PSScriptRoot 'evaluate_handoff_combined.ps1'
$comparer=Join-Path $PSScriptRoot 'compare_handoff_combined.py'
if(Test-Path -LiteralPath $out){throw 'Preserve existing combined evidence; choose a new RunTag'}
function Assert-Idle {
    $busy=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python.*|kit.*|isaac-sim.*)\.exe$' -and
        ($_.ExecutablePath -match '^E:[/\\]IsaacLab' -or $_.CommandLine -match 'E:[/\\]IsaacLab')})
    if($busy.Count){throw "Workspace Python/Kit active: $($busy.ProcessId -join ',')"}
}
Assert-Idle
$paths=@($PSCommandPath,$wrapper,$comparer,$roll,$stand,
    (Join-Path $PSScriptRoot 'evaluate_handoff_combined.py'),
    (Join-Path $PSScriptRoot 'evaluate_handoff_supported_startup.py'),
    (Join-Path $PSScriptRoot 'evaluate_handoff_mirror.py'),
    (Join-Path $PSScriptRoot 'audit_handoff_mirror_coordinates.py'),
    (Join-Path $PSScriptRoot 'compare_handoff_supported_startup.py'),
    (Join-Path $PSScriptRoot 'compare_handoff_transition_baseline.py'))
$paths+=@(Get-ChildItem -LiteralPath (Join-Path $root 'src/go2_recovery') -File -Filter '*.py' | ForEach-Object {$_.FullName})
$pins=@($paths | Sort-Object -Unique | ForEach-Object {@{path=$_;sha256=(Get-FileHash -LiteralPath $_).Hash}})
function Assert-Sources {
    foreach($p in $pins){if((Get-FileHash -LiteralPath $p.path).Hash -ne $p.sha256){throw "Frozen source changed: $($p.path)"}}
}
$null=New-Item -ItemType Directory -Path $out
$summary=[ordered]@{protocol='combined_12case_development_v1';status='running';started_at=(Get-Date -Format o);
    source_snapshot=$pins;rows=@();acceptance_eligible=$false;promotion_performed=$false;
    notice='Cold independent20-trial development cases. First9controls must match historical references before both factors ON. No new training or integrated locomotion acceptance.'}
$summaryPath=Join-Path $out 'summary.json'
function Save-Summary {$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryPath -Encoding utf8}
Save-Summary
$modes=@(@{s='off';m='off';label='s0m0'},@{s='supported';m='off';label='s1m0'},
         @{s='off';m='initial_right';label='s0m1'},@{s='supported';m='initial_right';label='s1m1'})
try {
    foreach($mode in $modes){foreach($pose in @('upright','side','upside_down')){
        Assert-Idle
        Assert-Sources
        $case=Join-Path $out "$($mode.label)-$pose"
        Write-Output "$(Get-Date -Format o) Begin $($mode.label)-$pose"
        $reused=$ReuseV1OffUpright -and $mode.label -eq 's0m0' -and $pose -eq 'upright'
        if($reused){
            $case=Join-Path $root 'evaluations/20260917-handoff-combined-v1/s0m0-upright'
            $invocation=Get-Content -LiteralPath (Join-Path $case 'invocation.json') -Raw | ConvertFrom-Json
            foreach($inputFile in $invocation.source_and_inputs){
                if((Get-FileHash -LiteralPath $inputFile.path).Hash -ne $inputFile.sha256){throw "Reused case input changed: $($inputFile.path)"}
            }
        } else {
            & $wrapper -Checkpoint $roll -StandCheckpoint $stand -OutputDir $case -Mode $mode.s -MirrorMode $mode.m -Pose $pose -Trials 20 -Seed 20260918
        }
        $candidate=Join-Path $case 'model_1999.pt_recovery_metrics.json'
        $auditName=if($reused){'exact_comparison_reaudit_v2.json'}else{'exact_comparison.json'}
        $audit=Join-Path $case $auditName
        Assert-Idle
        Assert-Sources
        & $python -B $comparer --candidate $candidate --output $audit *> (Join-Path $case "$auditName.log")
        if($LASTEXITCODE -ne 0){throw "Combined exact audit failed: $audit"}
        $r=Get-Content -LiteralPath $audit -Raw | ConvertFrom-Json
        if($r.exact_equivalence -ne $true -or $r.mismatch_count -ne 0){throw 'Exact comparison did not pass'}
        $summary.rows+=@{startup=$mode.s;mirror=$mode.m;pose=$pose;reused_simulation=[bool]$reused;final=$r.final_valid_stands;
            trials=20;report=$candidate;report_sha256=(Get-FileHash -LiteralPath $candidate).Hash;
            audit=$audit;checked_scalars=$r.checked_scalars;interface_audit=$r.interface_audit}
        Save-Summary
        Write-Output "PASS $($mode.label)-$pose final=$($r.final_valid_stands)/20 exact=$($r.checked_scalars)"
    }}
    $summary.status='completed_development_diagnostic'
} catch {
    $summary.status='error'
    $summary.error=$_.Exception.Message
    throw
} finally {
    $summary.finished_at=Get-Date -Format o
    Save-Summary
}
Write-Output "All12 cases complete: $summaryPath. Not promotion or fresh-generalization acceptance."
