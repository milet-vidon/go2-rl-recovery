# Lightweight wrapper regression: every valid invocation is forcibly DryRun.
# No simulator, Python, model, or temporary file is created.
$ErrorActionPreference = 'Stop'
$wrapper = Join-Path $PSScriptRoot 'train.ps1'
$bank = 'E:\IsaacLab\go2-rl-open-source\datasets\recovery_states\nominal_pd_v1_20260916'
if (-not (Test-Path -LiteralPath $bank)) { throw 'Expected existing E-drive bank fixture is missing.' }
$env:TEMP = 'E:\IsaacLab\tmp'; $env:TMP = 'E:\IsaacLab\tmp'
$checks = 0
function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Get-DryCommand([hashtable]$Parameters) {
    # The caller cannot remove this switch; the wrapper returns before isaaclab.bat.
    $lines = @(& $wrapper @Parameters -DryRun 6>&1 | ForEach-Object { "$_" })
    $commands = @($lines | Where-Object { $_.StartsWith('isaaclab.bat ') })
    Assert-True ($commands.Count -eq 1) 'DryRun did not return exactly one training command.'
    return $commands[0]
}
function Assert-Rejected([hashtable]$Parameters, [string]$Pattern) {
    $errorText = $null
    try { $null = Get-DryCommand $Parameters } catch { $errorText = $_.Exception.Message }
    Assert-True ($null -ne $errorText) 'Invalid parameter combination was accepted.'
    Assert-True ($errorText -match $Pattern) "Unexpected rejection: $errorText"
}
$base = @{Stage='recovery_bank_smith_nominal'; BankPath=$bank; NumEnvs=128; MaxIterations=2000;
          LoadRun='bootstrap_fresh42_target_v1'; Checkpoint='model_0.pt'; RunName='smith_weight_dry_test'; Headless=$true}
$baseline = Get-DryCommand $base
$expected = 'isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-v0 --num_envs 128 --max_iterations 2000 --run_name smith_weight_dry_test --device cuda:0 --rendering_mode performance --kit_args=--/app/vulkan=false --resume --load_run bootstrap_fresh42_target_v1 --checkpoint model_0.pt --headless'
Assert-True ($baseline -ceq $expected) 'Omitted parameter changed the original generated command.'
foreach ($weight in @(10, 30)) {
    $argsForWeight = $base.Clone(); $argsForWeight.SmithStandWeight = $weight
    $command = Get-DryCommand $argsForWeight
    Assert-True ($command -ceq "$baseline env.rewards.smith_stand.weight=$weight.0") 'Weight changed arguments other than the one stand-weight override.'
    Assert-True (([regex]::Matches($command, 'env\.rewards\.smith_stand\.weight=')).Count -eq 1) 'Duplicate stand override.'
}
foreach ($invalid in @(0, -10, 20, 30.5, 'nan', 'anything')) {
    $invalidArgs = $base.Clone(); $invalidArgs.SmithStandWeight = $invalid
    Assert-Rejected $invalidArgs 'SmithStandWeight'
}
# Cover every current non-Smith stage; rejection occurs before bank checks/env writes.
$tokens = $null; $parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($wrapper, [ref]$tokens, [ref]$parseErrors)
Assert-True ($parseErrors.Count -eq 0) 'Training wrapper has PowerShell parse errors.'
$stageParam = $ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq 'Stage' }
$validateSet = $stageParam.Attributes | Where-Object { $_.TypeName.FullName -eq 'ValidateSet' }
$stages = @($validateSet.PositionalArguments | ForEach-Object { $_.Value })
foreach ($stage in $stages | Where-Object { $_ -ne 'recovery_bank_smith_nominal' }) {
    Assert-Rejected @{Stage=$stage; SmithStandWeight=30} 'only supported by recovery_bank_smith_nominal'
}
$natural = Get-DryCommand @{Stage='natural_stop'; Headless=$true; NumEnvs=1; MaxIterations=2}
Assert-True (-not $natural.Contains('smith_stand')) 'Unrelated stage received Smith override.'
# Decimal output must remain Hydra-compatible under comma-decimal locales.
$priorCulture = [Threading.Thread]::CurrentThread.CurrentCulture
try {
    [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::GetCultureInfo('de-DE')
    $localeArgs = $base.Clone(); $localeArgs.SmithStandWeight = 30
    $localeCommand = Get-DryCommand $localeArgs
    Assert-True ($localeCommand.EndsWith('env.rewards.smith_stand.weight=30.0')) 'Override depends on locale decimal separator.'
} finally { [Threading.Thread]::CurrentThread.CurrentCulture = $priorCulture }
Write-Output "Passed $checks lightweight wrapper checks; all valid commands were DryRun; no simulator launched."
Write-Output "Generated example: $baseline env.rewards.smith_stand.weight=30.0"
