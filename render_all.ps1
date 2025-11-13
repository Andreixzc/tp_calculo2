<#
PowerShell helper to render all manim animations.

This script will attempt to activate the `myenv` virtualenv if present
and then invoke `render_all.py` with any supplied options.

Examples:
  .\render_all.ps1                # default pattern '*_animation.py', quality=qh
  .\render_all.ps1 -Pattern *.py -Quality ql -Workers 2
  .\render_all.ps1 -DryRun       # show commands but don't run
#>

param(
    [string]$Pattern = "*_animation.py",
    [string]$Quality = "qh",
    [int]$Workers = 1,
    [switch]$DryRun
)

$venvActivate = Join-Path -Path $PSScriptRoot -ChildPath "myenv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtualenv: $venvActivate"
    # dot-source to change current session environment
    & $venvActivate
} else {
    Write-Host "No virtualenv activation script found at: $venvActivate" -ForegroundColor Yellow
    Write-Host "If you have a venv, activate it before running this script to ensure manim is available."
}

$scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "render_all.py"
$argsList = @()
$argsList += "-p"; $argsList += $Pattern
$argsList += "-q"; $argsList += $Quality
$argsList += "-w"; $argsList += $Workers.ToString()
if ($DryRun) { $argsList += "--dry-run" }

Write-Host "Running: python $scriptPath $($argsList -join ' ')"
python $scriptPath @argsList
