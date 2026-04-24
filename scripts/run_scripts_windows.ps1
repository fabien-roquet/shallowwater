<#
Run shallowwater example scripts on Windows PowerShell.

Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\run_scripts_windows.ps1 -All
  powershell -ExecutionPolicy Bypass -File .\scripts\run_scripts_windows.ps1 -Script scripts\02_gravity_waves.py
  powershell -ExecutionPolicy Bypass -File .\scripts\run_scripts_windows.ps1 -List

From PowerShell 7+ you can also run:
  .\scripts\run_scripts_windows.ps1 -All
#>

param(
    [switch]$All,
    [switch]$List,
    [switch]$Python,
    [string]$Script
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

function Get-ExampleScripts {
    Get-ChildItem -Path "scripts" -Filter "*.py" -File |
        Where-Object { $_.Name -ne "strip_notebooks.py" } |
        Sort-Object FullName |
        ForEach-Object { $_.FullName }
}

function Run-PythonScript {
    param([string]$Path)

    Write-Host "==> Running $Path"
    if ($Python) {
        python $Path
    } else {
        uv run python $Path
    }
}

New-Item -ItemType Directory -Force -Path "animations" | Out-Null

if ($List) {
    Get-ExampleScripts | ForEach-Object {
        Resolve-Path -Relative $_
    }
    exit 0
}

if ($All) {
    Get-ExampleScripts | ForEach-Object {
        Run-PythonScript $_
    }
    exit 0
}

if (-not $Script) {
    Write-Error "Please provide -Script <path> or use -All."
    exit 2
}

if (-not (Test-Path $Script -PathType Leaf)) {
    Write-Error "Script not found: $Script"
    exit 1
}

Run-PythonScript $Script
