param(
    [string]$ProjectRoot = $env:ALLEGRO_PROJECT_ROOT,
    [string]$IsaacPython = $env:ISAACLAB_PYTHON_BAT,
    [int]$NumEnvs = 2048,
    [long]$TotalEnvSteps = 1000000000
)

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = (Resolve-Path ".").Path }
if ([string]::IsNullOrWhiteSpace($IsaacPython)) { throw "Set ISAACLAB_PYTHON_BAT to IsaacLab _isaac_sim\python.bat" }

Set-Location $ProjectRoot
& $IsaacPython "src\allegro_rl\tasks\task2\task2_train.py" `
    --num-envs $NumEnvs `
    --total-env-steps $TotalEnvSteps `
    --rollouts 64 `
    --learning-epochs 5 `
    --mini-batches 8 `
    --summary-interval 10 `
    --save-freq-env-steps 20000000 `
    --headless `
    --device cuda:0
