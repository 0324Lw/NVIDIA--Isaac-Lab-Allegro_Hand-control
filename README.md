# 🖐️ 基于 NVIDIA Isaac Lab 的 Allegro Hand 灵巧手纯 RL 控制项目
 
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-brightgreen)
![skrl](https://img.shields.io/badge/RL-skrl%20PPO%20%7C%20RMA--Ready-purple)
![OS](https://img.shields.io/badge/OS-Ubuntu%20%7C%20Windows-green)
 
本项目是一个基于 NVIDIA Isaac Lab 的 Allegro Hand 灵巧手强化学习控制项目。项目包含 4 个递进任务：基础手势位姿追踪、掌内物体重定向、动态抓取与工具使用、Blind Sim2Real / RMA 鲁棒重定向训练。
 
这个仓库是我在学习灵巧手强化学习控制过程中整理出来的一版纯 RL baseline。和人形机器人复杂舞蹈、武术动作相比，灵巧手的部分任务可以更直接地通过纯强化学习和合理的奖励设计进行探索；但是如果想实现非常高质量、可泛化、可迁移的真实灵巧操作，专业路线仍然通常需要示教数据、遥操作数据、动作重定向、模仿学习、触觉反馈、视觉感知和 Sim2Real 体系。因此，这个仓库更适合作为一个早期探索版、学习版、纯 RL baseline 保存下来，希望能为在学习 Isaac Lab、灵巧手控制、接触丰富操作和强化学习的同学提供一个可以参考、可以运行、可以继续修改的基础工程。项目重点是尽量把每个任务的环境、测试、训练、评估、数据生成和日志拆清楚。代码中仍然有很多可以继续改进的地方，欢迎大家根据自己的 Isaac Lab 版本、显卡配置和研究目标继续修改。
 
---
 
## 🎬 训练效果展示
 
| Scene | Preview |
|---|---|
| 手势位姿追踪 / 掌内物体重定向 | ![Allegro pose tracking demo](assets/gifs/allegro_pose_tracking_demo.gif) |
| 动态抓取 / Sim2Real 鲁棒重定向 | ![Allegro sim2real demo](assets/gifs/allegro_sim2real_demo.gif) |
 
---
 
## ✨ 项目特点
 
- 基于 NVIDIA Isaac Lab 和 Allegro Hand 灵巧手资产。
- 包含 4 个递进任务，从无物体手势追踪到掌内重定向、动态抓取和 Blind Sim2Real / RMA 鲁棒训练。
- Task1 / Task2 / Task3 / Task4 均使用 `skrl` PPO 训练流程；脚本名称带 `skrl` 的训练与测试脚本均应真实调用 `skrl` 相关接口。
- Task4 当前实现为 Teacher PPO 阶段：Actor 使用 `teacher_obs`，Critic 使用 `privileged_obs`，同时保留 `blind_obs` 和 `history_obs`，为后续 Student / Adapter / RMA 蒸馏准备。
- 所有 Allegro Hand 任务环境代码独立实现，不依赖其他任务环境继承，避免任务之间互相污染。
- 每个任务提供独立环境测试、训练脚本和模型测试脚本。
- 使用 `tqdm` 风格训练进度条，方便实时查看训练速度、奖励、掉落率、接触数量、目标误差和关键遥测指标。
 
---
 
## 📁 项目结构
 
```text
allegro_hand_isaaclab_rl/
├── assets/
│   ├── gifs/
│   ├── motions/
│   └── usd/
├── configs/
│   ├── task1_pose_tracking.yaml
│   ├── task2_inhand_reorientation.yaml
│   ├── task3_dynamic_grasp_tool_use.yaml
│   └── task4_blind_sim2real_rma.yaml
├── src/
│   └── allegro_rl/
│       ├── common/
│       │   ├── allegro_skrl_models.py
│       │   ├── allegro_skrl_wrappers.py
│       │   ├── eval_curriculum_utils.py
│       │   ├── info_utils.py
│       │   ├── model_eval_utils.py
│       │   ├── paths.py
│       │   ├── progress.py
│       │   ├── running_mean_std.py
│       │   ├── skrl_models.py
│       │   └── vec_wrappers.py
│       ├── data/
│       │   └── generate_task1_pose_dataset.py
│       └── tasks/
│           ├── task1/
│           │   ├── task1_config.py
│           │   ├── task1_env.py
│           │   ├── task1_train.py
│           │   └── task1_model_test.py
│           ├── task2/
│           │   ├── task2_config.py
│           │   ├── task2_env.py
│           │   ├── task2_scene.py
│           │   ├── task2_train.py
│           │   └── task2_model_test.py
│           ├── task3/
│           │   ├── task3_config.py
│           │   ├── task3_env.py
│           │   ├── task3_scene.py
│           │   ├── task3_train.py
│           │   └── task3_model_test.py
│           └── task4/
│               ├── task4_config.py
│               ├── task4_env.py
│               ├── task4_scene.py
│               ├── task4_train.py
│               └── task4_model_test.py
├── tests/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── scripts/
│   ├── ubuntu/
│   └── windows/
├── logs/
├── docs/
├── LICENSE
└── README.md
```
 
| 目录 | 说明 |
|---|---|
| `configs/` | 每个任务的配置文件，便于统一管理任务参数。 |
| `src/allegro_rl/common/` | 通用网络模型、评估工具、日志工具、路径工具、归一化工具和 wrapper 占位。 |
| `src/allegro_rl/data/` | 数据生成脚本，例如 Task1 随机目标手势 / 目标关节位姿数据集。 |
| `src/allegro_rl/tasks/taskX/` | 每个任务的配置、环境、场景、训练脚本和模型测试脚本。 |
| `tests/` | 每个任务的环境测试脚本。 |
| `scripts/ubuntu/` | Ubuntu 下的测试、训练、评估和可视化脚本。 |
| `scripts/windows/` | Windows 下的训练与评估脚本模板。 |
| `logs/` | 默认训练日志和 checkpoint 输出目录。 |
| `assets/` | README 图片、GIF、USD 占位和 motion / dataset 文件。 |
| `docs/` | 项目说明、任务设计、训练说明和 troubleshooting 文档。 |
 
---
 
## 🛠️ 建议硬件与系统配置
 
### 最低测试配置
 
用于环境测试、smoke training、低并发调试和模型测试：
 
- Ubuntu 22.04 / 24.04
- NVIDIA GPU，建议显存 16GB 以上
- Python 3.11
- PyTorch 2.x
- Isaac Sim / Isaac Lab
- `skrl`, `tensorboard`, `tqdm`, `numpy`
 
在显存较小的设备上，建议从很小的并发开始：
 
```bash
--num-envs 1
--num-envs 4
--num-envs 8
--num-envs 16
```
 
### 推荐训练配置
 
用于较大规模训练和长时间实验：
 
- NVIDIA RTX 3090 / 4090 或同级别 GPU
- 显存 24GB 或更高
- Ubuntu 环境优先
- Isaac Lab 环境能够稳定运行
 
较大显存设备可以逐步尝试：
 
```bash
--num-envs 512
--num-envs 1024
--num-envs 2048
--num-envs 4096
```
 
灵巧手任务虽然没有人形机器人那样复杂的全身稳定问题，但接触丰富、物体掉落、接触传感器、掌内重定向和 Sim2Real 随机化仍然容易引入训练震荡。不要一开始直接使用最大并发，建议先运行环境测试和 smoke training。
 
---
 
## 🚀 基础准备
 
### 1. 安装 Isaac Lab 环境
 
请先按照 NVIDIA Isaac Lab 官方文档安装 Isaac Sim / Isaac Lab，并确认 Isaac Lab 的 Python 环境可以正常导入：
 
```bash
python -c "import isaaclab; print('isaaclab ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
 
### 2. 克隆项目
 
```bash
git clone https://github.com/0324Lw/NVIDIA--Isaac-Lab-Allegro-Hand-control.git allegro_hand_isaaclab_rl
cd allegro_hand_isaaclab_rl
```
 
如果你的仓库名称不同，请把上面的 GitHub 地址替换为你自己的仓库地址。
 
### 3. 设置 PYTHONPATH
 
```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```
 
也可以直接使用 `scripts/ubuntu/` 下的脚本，这些脚本会自动设置项目路径。
 
### 4. 生成或检查 Task1 目标位姿数据
 
Task1 使用随机目标关节位姿数据集：
 
```text
assets/motions/task1_target_poses.pt
```
 
如果仓库中没有这个文件，可以使用数据生成脚本重新生成：
 
```bash
bash scripts/ubuntu/generate_task1_dataset.sh
```
 
说明：
 
- `task1_target_poses.pt` 用于 Task1 手势 / 关节位姿追踪。
- Task2 / Task3 / Task4 主要在环境内采样目标和物体状态，不依赖大规模动捕数据。
- 如果后续引入遥操作、动捕、示教或视觉恢复数据，建议统一放在 `assets/motions/` 或外部数据目录中，并避免把大文件直接上传到 GitHub。
 
### 5. 安装 Python 依赖
 
在 Isaac Lab 对应的 Python 环境中安装必要依赖：
 
```bash
pip install skrl tensorboard tqdm numpy
```
 
如果你的 Isaac Lab 环境已经包含部分依赖，可以按需跳过。
 
---
 
## ⚡ 快速开始
 
### 1. 环境测试
 
建议先从 Task1 开始测试，再进入后续任务。
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/test_task4_env.sh
```
 
如果显存不足，可以打开对应脚本，降低 `--num-envs`。
 
### 2. Smoke 训练
 
Smoke training 只用于确认训练管线可以启动、日志可以写入、checkpoint 可以保存，不用于评估最终效果。
 
```bash
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task4_skrl_smoke.sh
```
 
### 3. 模型测试
 
训练完成后，可以使用 eval 脚本加载 checkpoint 做推理测试。
 
```bash
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/allegro_task1_model.pt 1.0
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/allegro_task2_model.pt 1.0
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/allegro_task3_model.pt 1.0
bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run_name>/final_checkpoint/allegro_task4_teacher_model.pt 1.0
```
 
### 4. GUI 可视化测试
 
训练完成后，可以使用可视化脚本打开 Isaac Sim GUI 查看策略效果：
 
```bash
bash scripts/ubuntu/visual/visualize_task1.sh logs/task1/<run_name>/final_checkpoint 1.0
bash scripts/ubuntu/visual/visualize_task2.sh logs/task2/<run_name>/final_checkpoint 1.0
bash scripts/ubuntu/visual/visualize_task3.sh logs/task3/<run_name>/final_checkpoint 1.0
bash scripts/ubuntu/visual/visualize_task4.sh logs/task4/<run_name>/final_checkpoint 1.0
```
 
---
 
## 🧩 任务设计总览
 
| Task | 目标 | 环境特点 | 训练重点 | 主要脚本 |
|---|---|---|---|---|
| Task1 | 手势 / 关节位姿追踪 | 无外部物体，目标关节位姿数据集，16 DoF 控制 | 稳定快速跟踪目标手势，降低关节误差和抖动 | `task1_env.py`, `task1_train.py`, `task1_model_test.py` |
| Task2 | 掌内物体重定向 | cube / sphere，接触传感器，privileged critic | 保持物体不掉落，降低目标四元数误差 | `task2_env.py`, `task2_scene.py`, `task2_train.py`, `task2_model_test.py` |
| Task3 | 动态抓取与工具使用 | tabletop pen / cup，floating base，22 维动作 | 接近、接触、抓取、抬升和姿态调整 | `task3_env.py`, `task3_scene.py`, `task3_train.py`, `task3_model_test.py` |
| Task4 | Blind Sim2Real / RMA 鲁棒重定向 | blind obs、teacher obs、privileged obs、history obs、DR | Full DR 下保持物体、重定向和扰动恢复；为 Student / Adapter 准备 | `task4_env.py`, `task4_scene.py`, `task4_train.py`, `task4_model_test.py` |
 
---
 
## ➡️ Task 1：手势 / 关节位姿追踪
 
Task1 是最基础的灵巧手控制任务，用于让 Allegro Hand 在没有外部物体的情况下跟踪随机目标关节位姿，例如握拳、张开、捏合、伸指等。
 
### 任务目标
 
- Allegro Hand 跟踪给定的 16 维目标关节位姿。
- 学习稳定、快速、低抖动的关节控制策略。
- 作为后续接触操作任务的基础控制 baseline。
- 保持 16 DoF 动作空间和 64 维单帧观测结构。
 
### 环境设计
 
- 使用 Isaac Lab 中的 Allegro Hand 资产。
- 动作输出为 16 个受控关节的目标位置残差。
- 单帧 actor observation 为 64 维，包含 `target_error`、`joint_pos`、`joint_vel`、`last_filtered_action`。
- 5 帧堆叠后 policy input 为 320 维。
- 训练代码使用 `skrl` PPO。
 
### 常用命令
 
```bash
bash scripts/ubuntu/generate_task1_dataset.sh
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task1_skrl_laptop.sh
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/allegro_task1_model.pt 1.0
```
 
### 训练时重点观察
 
- `Pose_Error_Rad` 或目标关节误差是否下降。
- `Joint_Velocity` 是否过大。
- `Curriculum_Progress_K` 是否平滑推进。
- `R_Track`、`R_Smooth`、`P_ActionRate` 是否稳定。
- PPO 的 `approx_kl`、`clip_fraction`、学习率是否正常。
 
---
 
## ➡️ Task 2：掌内物体重定向
 
Task2 在 Task1 的基础上引入掌内物体和接触操作，目标是让 Allegro Hand 抓住 cube 或 sphere，并将物体重定向到目标四元数。
 
### 任务目标
 
- 保持物体在掌内不掉落。
- 通过手指接触和关节控制降低目标姿态误差。
- 学习接触保持、预掉落恢复和掌内微调。
- 为 Task4 的 Blind Sim2Real 鲁棒重定向打基础。
 
### 环境设计
 
- 使用独立 `task2_scene.py` 构建 Allegro Hand、cube、sphere 和 fingertip contact sensor。
- Actor observation 和 privileged observation 分离。
- Actor 看常规本体感觉和目标信息；Critic 可看物体质量、摩擦、COM 偏移等 privileged 信息。
- 使用 geodesic quaternion error 评估物体姿态误差。
- 使用 `skrl` PPO 训练，采用非对称 Actor-Critic。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_laptop.sh
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/allegro_task2_model.pt 1.0
```
 
### 训练时重点观察
 
- `Object_Height` 是否保持在安全高度以上。
- `Drop_Rate` 是否逐步下降。
- `Active_Contacts` 或 `Contact_Count` 是否合理。
- `Geodesic_Error_Rad` 是否下降。
- `P_PreDrop` 是否过大。
- PPO 的 KL、clip fraction 和 value loss 是否稳定。
 
---
 
## ➡️ Task 3：动态抓取与工具使用
 
Task3 进一步引入桌面工具和浮动基座控制，让 Allegro Hand 学习从桌面附近接近、接触、抓取、抬升并调整 pen / cup 等工具的姿态。
 
### 任务目标
 
- 控制 Allegro Hand 的 16 个手指关节和 6 维 floating base。
- 从桌面附近接近目标工具。
- 形成有效接触和 force-closure 风格抓取。
- 抬升物体并尝试进行姿态调整。
 
### 环境设计
 
Task3 是独立环境，不继承 Task1 / Task2：
 
- action dimension：22，其中 16 维 hand action + 6 维 floating base action。
- actor observation：147 维。
- privileged observation：168 维。
- 5 帧堆叠后 Actor input 为 735 维，Critic input 为 840 维。
- 使用 `task3_scene.py` 构建 table、pen、cup、Allegro Hand 和 hand contact sensor。
- 奖励包括 approach、pregrasp、contact、force closure、grip、lift、orientation、stability 和 workspace penalty。
- 使用 `skrl` PPO 训练，采用非对称 Actor-Critic。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_laptop.sh
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/allegro_task3_model.pt 1.0
```
 
### 训练时重点观察
 
- `TCP_Dist` 或 `TCP_Pregrasp_Dist` 是否下降。
- `SoftContact_Count` / `HardContact_Count` 是否上升到合理范围。
- `Lift` 和 `Obj_H` 是否逐步提高。
- `Drop`、`SlideOut`、`TableCrash` 是否过高。
- `SO3_Err` 是否随训练下降。
- `Base_H`、`Base_XY` 和 `WorkspaceClamp` 是否异常。
 
Task3 的动作效果不应被理解为成熟的通用工具使用能力。它是一个纯 RL 动态抓取 baseline，用于学习接触丰富任务、浮动基座控制和奖励设计的难点。
 
---
 
## ➡️ Task 4：Blind Sim2Real / RMA 鲁棒重定向
 
Task4 面向 Blind Sim2Real 和 RMA 风格鲁棒重定向。Student 视角只依赖 blind obs 和 history obs；当前训练脚本先实现 Teacher PPO，Actor 使用 teacher obs，Critic 使用 privileged obs，同时保存 RMA-ready checkpoint，为后续 Student / Adapter 蒸馏做准备。
 
### 任务目标
 
- 在重度域随机化下保持掌内物体并降低姿态误差。
- 引入动作延迟、死区、电机效率、关节刚度 / 阻尼、触觉 dropout、状态 dropout 和外部扰动。
- 为后续 Blind Student / RMA Adapter 学习准备 teacher policy、blind obs 和 history obs。
- 测试灵巧手纯 RL 在 Sim2Real 随机化下的稳定性上限。
 
### 环境设计
 
Task4 是独立环境，不继承 Task1 / Task2 / Task3：
 
- action dimension：16。
- blind student observation：108 维。
- teacher observation：139 维。
- privileged observation：206 维。
- history frame：104 维。
- history length：50。
- history observation：5200 维。
- 当前 Teacher PPO：Actor 输入 `teacher_obs × 5`，Critic 输入 `privileged_obs × 5`。
 
Sim2Real 随机化包括：
 
- mass randomization
- friction randomization
- object scale randomization
- COM offset randomization
- inertia scale randomization
- joint efficiency randomization
- joint stiffness / damping randomization
- action delay
- actuator deadzone
- action noise
- tactile noise / tactile dropout
- joint state noise / state dropout
- slip disturbance
- external push disturbance
 
训练后会保存：
 
```text
allegro_task4_skrl_agent.pt
teacher_model.pt
allegro_task4_teacher_model.pt
task4_train_metadata.pt
```
 
其中 `allegro_task4_teacher_model.pt` 是当前模型测试优先使用的 Teacher eval checkpoint。后续如果继续扩展 Student / Adapter，可基于 `blind_obs` 和 `history_obs` 做蒸馏或 RMA-style 适应模块。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task4_env.sh
bash scripts/ubuntu/train_task4_skrl_smoke.sh
bash scripts/ubuntu/train_task4_skrl_laptop.sh
bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run_name>/final_checkpoint/allegro_task4_teacher_model.pt 1.0
```
 
### 训练时重点观察
 
- `DR_K`
- `Reward_K`
- `SO3_Error`
- `Object_Height`
- `Contact_Count`
- `Drop`
- `Success`
- `ActionDelay`
- `Deadzone`
- `JointEfficiency`
- `TactileDropout`
- `StateDropout`
- `DisturbanceNorm`
- PPO 的 `approx_kl`、loss、learning rate
 
---
 
## 📊 日志与模型保存
 
训练日志默认保存在：
 
```text
logs/task1/
logs/task2/
logs/task3/
logs/task4/
```
 
每个训练 run 通常包含：
 
```text
checkpoint_<env_steps>/
final_checkpoint/
train_metadata.pt
```
 
Task4 Teacher 还会额外保存：
 
```text
allegro_task4_skrl_agent.pt
teacher_model.pt
allegro_task4_teacher_model.pt
task4_train_metadata.pt
```
 
可以使用 TensorBoard 查看训练过程：
 
```bash
tensorboard --logdir logs
```
 
训练过程中会记录以下类型的信息：
 
- `reward_components`：各奖励项。
- `events`：drop、success、timeout、table crash、workspace clamp 等事件。
- `telemetry`：目标误差、物体高度、接触数量、课程阶段、DR 参数等训练指标。
- `debug`：观测维度、reward 范围、异常值检查等。
- `ppo`：PPO 更新信息、KL、loss、学习率等。
- `rma`：Task4 中 blind obs / history obs 的辅助记录，为后续 Student / Adapter 使用。
 
---
 
## 💻 Ubuntu 使用说明
 
当前仓库以 Ubuntu / Isaac Lab 环境为主。常用脚本在：
 
```text
scripts/ubuntu/
```
 
推荐顺序是：
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/allegro_task1_model.pt 1.0
```
 
后续任务同理，先测试环境，再 smoke training，再进行长训练和模型测试。
 
Windows 脚本模板位于：
 
```text
scripts/windows/
```
 
如果你在 Windows 上运行，需要根据本机的 Isaac Lab 路径、Python 路径、项目路径和显卡状态修改脚本。建议先运行 `check_task*_windows_ready.ps1`，再运行 smoke training。
 
---
 
## 🧭 推荐训练顺序
 
推荐顺序：
 
1. 先生成 Task1 目标位姿数据集。
2. 训练 Task1，获得稳定的手势 / 关节位姿追踪 checkpoint。
3. 训练 Task2，学习掌内物体重定向。
4. 训练 Task3，探索桌面动态抓取、抬升和工具使用。
5. 训练 Task4 Teacher PPO，建立 Blind Sim2Real / RMA-ready teacher checkpoint。
6. 后续可在 Task4 基础上继续扩展 Student / Adapter / RMA 蒸馏和部署接口。
 
也可以每个任务从零开始训练，但灵巧手接触操作对 reset、接触传感器、奖励项、掉落终止和动作平滑非常敏感。建议先用较小并发完成环境测试和 smoke training，再进行长时间训练。
 
---
 
## 📌 当前状态与限制
 
- 本项目主要用于学习、复现实验和开源交流。
- 当前代码完成了四个任务的 Isaac Lab 环境、环境测试、训练脚本和模型测试脚本。
- 这个仓库是 pure-RL baseline，不代表工业级灵巧手操作最终路线。
- 高质量灵巧操作通常还需要示教数据、遥操作数据、触觉反馈、视觉感知、模仿学习和更完整的 Sim2Real 流程。
- Task3 的动态抓取和工具使用是学习版任务，不等同于通用工具操作能力。
- Task4 当前是 Teacher PPO 阶段，已经保留 `blind_obs` 和 `history_obs`，但还没有完成最终 Student / Adapter 部署策略。
- 不同 Isaac Lab / Isaac Sim 版本之间可能存在 API 差异，需要根据本地环境做少量适配。
- 训练效果会受到 GPU、并发数、随机种子、训练步数和超参数影响。
- 本项目不是官方 Allegro Hand、NVIDIA 或 Wonik Robotics 项目，只是个人学习和开源整理。
 
---
 
## ❓ 常见问题
 
### 1. `ModuleNotFoundError: No module named torch`
 
通常是没有进入 Isaac Lab 对应的 Python / conda 环境。请先确认：
 
```bash
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
 
### 2. Isaac Lab / `pxr` 导入报错
 
涉及 Isaac Lab、USD、`pxr` 的文件需要在 Isaac Sim / Isaac Lab 环境中运行。测试脚本中如果需要 AppLauncher，应保证先启动 AppLauncher，再导入依赖 Isaac Lab 的环境文件。
 
### 3. 训练启动后显存不足怎么办？
 
先降低并发数：
 
```bash
--num-envs 1
--num-envs 4
--num-envs 8
--num-envs 16
--num-envs 32
```
 
确认能跑通后再逐步增加。
 
### 4. Smoke training 的效果不好正常吗？
 
正常。Smoke training 只用于检查训练流程是否能启动和保存模型，不代表最终策略效果。
 
### 5. 为什么灵巧手任务容易掉物体？
 
灵巧手任务是接触丰富任务，物体稳定性依赖接触位置、摩擦、手指闭合策略、动作平滑、姿态误差和奖励设计。早期随机策略或 smoke checkpoint 掉物体是正常现象，需要通过长期训练和合理课程逐步改善。
 
### 6. 为什么模型测试脚本不使用 `agent.act()`？
 
为了避免不同 `skrl` 版本在测试阶段出现 `agent.act()` 卡住或接口不兼容的问题，模型测试脚本直接加载 eval checkpoint 中的 policy 权重，并使用 deterministic direct policy forward。
 
### 7. 为什么 Task4 当前叫 RMA-ready，而不是完整 RMA Student？
 
当前 Task4 已经保留 blind obs、history obs、teacher obs 和 privileged obs，并使用 Teacher PPO 训练可供蒸馏的 teacher policy。但完整 Student / Adapter / RMA 部署策略还需要后续单独扩展，因此当前更准确地称为 RMA-ready Teacher PPO。
 
### 8. 为什么要先跑环境测试？
 
灵巧手训练中的很多问题不是 PPO 本身造成的，而是 reset、观测维度、关节映射、接触检测、目标四元数、掉落终止、动作延迟或奖励项有问题。先跑测试可以减少后续训练调参的时间。
 
### 9. 这个项目能直接真机部署吗？
 
不能直接保证。这个仓库目前是 Isaac Lab 仿真学习 baseline。真机部署还需要安全限幅、低层控制接口、状态估计、延迟测试、动力学参数校准、触觉 / 视觉接口、实机保护逻辑和大量 Sim2Real 验证。
 
---
 
## 📄 License
 
This project is released under the MIT License.
 
See the `LICENSE` file for details.
 
---
 
## 🙏 Acknowledgements
 
感谢以下开源项目和工具：
 
- NVIDIA Isaac Sim / Isaac Lab
- Allegro Hand robot asset and related documentation
- PyTorch
- skrl reinforcement learning library
- TensorBoard
- tqdm
- 机器人强化学习、灵巧手控制、接触丰富操作和 Isaac Lab 开源社区
 
如果这个项目对你有帮助，欢迎参考、修改和继续完善。也欢迎指出代码或文档中的问题。
 
联系邮箱：2559906288@qq.com  
小红书账号：574661219
