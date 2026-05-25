param(
    [string]$ProjectRoot = $env:ALLEGRO_PROJECT_ROOT,
    [string]$IsaacPython = $env:ISAACLAB_PYTHON_BAT
)

Write-Host "============================================================"
Write-Host "Allegro Hand Task2 Windows readiness check"
Write-Host "============================================================"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path ".").Path
}
if (!(Test-Path $ProjectRoot)) {
    throw "ProjectRoot not found: $ProjectRoot"
}

Write-Host "ProjectRoot = $ProjectRoot"

$TaskFile = Join-Path $ProjectRoot "src\allegro_rl\tasks\task2\task2_train.py"
if (!(Test-Path $TaskFile)) {
    throw "Missing task train file: $TaskFile"
}

if ([string]::IsNullOrWhiteSpace($IsaacPython)) {
    Write-Warning "ISAACLAB_PYTHON_BAT is not set. Set it before real Windows training."
} elseif (!(Test-Path $IsaacPython)) {
    throw "IsaacLab python.bat not found: $IsaacPython"
} else {
    Write-Host "IsaacPython = $IsaacPython"
}

Write-Host "[OK] Task2 Windows framework check completed."
