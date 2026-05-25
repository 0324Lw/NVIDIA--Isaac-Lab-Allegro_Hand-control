param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$ProjectRoot = $env:ALLEGRO_PROJECT_ROOT,
    [string]$IsaacPython = $env:ISAACLAB_PYTHON_BAT,
    [int]$NumEnvs = 4,
    [int]$Steps = 200,
    [double]$StartK = 1.0
)

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = (Resolve-Path ".").Path }
if ([string]::IsNullOrWhiteSpace($IsaacPython)) { throw "Set ISAACLAB_PYTHON_BAT to IsaacLab _isaac_sim\python.bat" }

Set-Location $ProjectRoot
& $IsaacPython "src\allegro_rl\tasks\task2\task2_model_test.py" `
    --checkpoint $Checkpoint `
    --num-envs $NumEnvs `
    --steps $Steps `
    --start-k $StartK `
    --print-interval 20 `
    --headless `
    --device cuda:0
