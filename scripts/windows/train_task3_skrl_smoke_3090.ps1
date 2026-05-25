param(
    [string]$ProjectRoot = $env:ALLEGRO_PROJECT_ROOT,
    [string]$IsaacPython = $env:ISAACLAB_PYTHON_BAT
)

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = (Resolve-Path ".").Path }
if ([string]::IsNullOrWhiteSpace($IsaacPython)) { throw "Set ISAACLAB_PYTHON_BAT to IsaacLab _isaac_sim\python.bat" }

Set-Location $ProjectRoot
& $IsaacPython "src\allegro_rl\tasks\task3\task3_train.py" `
    --num-envs 512 `
    --total-env-steps 5120 `
    --rollouts 4 `
    --learning-epochs 2 `
    --mini-batches 2 `
    --summary-interval 1 `
    --save-freq-env-steps 5120 `
    --headless `
    --device cuda:0
