# Go2 FootStand Student Policy Deployment Settings

本文记录当前 `conf/ppo/student_finetune.yaml` 组合 `task=go2_footstand/mujoco`
后，和真机部署直接相关的 student policy 设置。

对应训练入口：

```bash
uv run python scripts/train_rsl_rl_student_finetune.py
```

导出的部署模型在训练 run 目录下：

```text
policy.onnx
policy.pt
```

这两个导出模型都是 deterministic policy，即输出 Gaussian mean，不采样。

## Policy

Student actor:

```text
input_dim = 630
hidden_dims = [512, 256, 128]
activation = ELU
output_dim = 12
```

结构：

```text
630 -> 512 -> 256 -> 128 -> 12
```

Observation normalization 开启，并且已经包含在导出的 `policy.onnx` / `policy.pt`
里。部署时如果直接使用导出模型，不要再手动额外做一层 normalization。

训练时的 Gaussian exploration 设置：

```text
init_std = 0.5
std_type = scalar
```

这是训练用的 exploration 参数。部署导出模型输出 deterministic action mean，不使用随机采样。

## Student Observation

环境原始 `obs` 每帧是 45 维：

```text
linvel(3)
gyro(3)
local_gravity(3)
dof_pos - default_angles(12)
dof_vel(12)
last_action(12)
```

Student policy 会从每帧中删除前 3 维 `linvel`：

```text
student_frame_dim = 45
student_drop_start = 0
student_drop_dim = 3
```

所以 student policy 实际每帧输入是 42 维：

```text
gyro(3)
local_gravity(3)
dof_pos - default_angles(12)
dof_vel(12)
last_action(12)
```

历史长度：

```text
obs_history_len = 15
student_obs_dim = 15 * 42 = 630
```

History 是 FIFO，最新一帧放在最后。reset 时当前环境会把 reset frame 复制到
15 帧 history 中。真机部署建议同样初始化整段 history，不要用全 0 history 直接启动。

部署模型输入就是一个 630 维 float tensor：

```text
[frame_0_without_linvel, ..., frame_14_without_linvel]
```

其中 `frame_14` 是最新一帧。

## Observation Sources

`gyro`:

```text
base body angular velocity in body frame, dim=3
```

`local_gravity`:

```text
world gravity [0, 0, -1] transformed into base local frame, dim=3
```

`dof_pos - default_angles`:

```text
joint position error, dim=12
```

`dof_vel`:

```text
joint velocity, dim=12
```

`last_action`:

```text
previous policy action, dim=12
```

注意：`last_action` 是 policy 输出 action 的顺序，不是 `dof_pos/dof_vel` 的顺序。

部署时不要给 student actor 输入 `linvel`。`linvel` 只存在于训练时的 teacher/full obs
和 critic obs 中。

## Joint Order

`dof_pos` / `dof_vel` / `default_angles` 使用 MuJoCo joint 顺序：

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

当前 `default_angles`：

```text
[0.0, 0.8, -1.5,
 0.0, 0.8, -1.5,
 0.0, 1.0, -1.5,
 0.0, 1.0, -1.5]
```

## Action Order

Policy 输出 action 是 12 维，使用 MuJoCo actuator/control 顺序：

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

环境里用下面的 index 在 DOF 顺序和 control 顺序之间转换：

```python
_GO2_DOF_TO_CTRL = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
```

含义是：

```text
control_order_values = dof_order_values[_GO2_DOF_TO_CTRL]
```

如果真机 SDK 的电机顺序和这里不同，必须显式重排 action 和 joint state。

## Action To Motor Target

FootStand 当前不是普通的：

```text
target = default_angles + action_scale * action
```

而是增量式 motor target：

```text
action_clipped = clip(action, -1, 1)
motor_targets += action_clipped * 0.3
motor_targets = clip_to_joint_limits(motor_targets)
```

相关参数：

```text
clip_actions = 1.0
action_scale = 0.3
simulate_action_latency = false
```

`motor_targets` 的顺序是 action/control 顺序：

```text
FR, FL, RR, RL
```

真机部署时需要维护 persistent `motor_targets`。建议启动或 reset 时：

```text
motor_targets = current_joint_pos converted from joint order to action/control order
```

每个 control tick 更新一次：

```text
action = policy(obs_history)
action = clip(action, -1, 1)
motor_targets = motor_targets + 0.3 * action
motor_targets = clip_to_joint_limits(motor_targets)
send_position_targets(motor_targets)
```

同时把本 tick 的 policy action 保存为下一帧 observation 的 `last_action`。

## Control Frequency And PD

当前配置：

```text
sim_dt = 0.004
ctrl_dt = 0.02
policy/control frequency = 50 Hz
Kp = 35.0
Kd = 0.5
```

真机侧应按 50 Hz 更新 policy 和 position target。若真机底层控制器已有自己的
PD/阻尼设置，需要确认等效于训练中的 `Kp=35.0, Kd=0.5`。

## Deployment Inputs To Exclude

下面这些只用于训练，不应作为 student policy 的真机输入：

```text
critic obs: 724 dims
teacher policy obs: 675 dims, includes linvel
reward
domain randomization
observation noise
Gaussian exploration std
```

Teacher regularization 当前训练配置：

```text
teacher obs dim = 675
teacher has linvel = true
action_loss_coef = 0.05
kl_loss_coef = 0.01
```

这只影响训练 loss，不影响导出的 student policy 输入维度。

## Training Noise

训练时 observation noise：

```text
level = 1.0
scale_joint_angle = 0.01
scale_joint_vel = 1.5
scale_gyro = 0.2
scale_gravity = 0.05
scale_linvel = 0.1
```

noise 形式：

```text
x_noisy = x + uniform(-1, 1) * level * scale
```

真机部署时不要主动加这层随机 noise。

## Domain Randomization

训练时 DR：

```text
randomize_floor_friction = true
floor_friction_range = [0.4, 1.0]

randomize_link_mass = true
link_mass_scale_range = [0.9, 1.1]
torso_added_mass_range = [-1.0, 1.0]

randomize_torso_com = true
torso_com_offset_range = [-0.05, 0.05]

randomize_dof_armature = true
dof_armature_scale_range = [1.0, 1.05]

randomize_reset_joint_qpos = true
reset_joint_qpos_range = [-0.05, 0.05]
```

FootStand 中这些 DR 是 disabled：

```text
randomize_kp = false
randomize_kd = false
randomize_base_mass = false
random_com = false
push_robots = false
```

## Asset Note

当前 Go2 XML 中 `base3` collision geom 已注释：

```xml
<!-- <geom name="base3" size="0.047" pos="0.293 0 -0.06" class="collision"/> -->
```

对应的 `base3_contact` sensor 引用也已移除。这个改动是为了匹配真机已拆掉该部件的状态。

## Source Files

主要来源：

```text
conf/ppo/student_finetune.yaml
conf/ppo/task/go2_footstand/mujoco.yaml
src/unilab/envs/locomotion/go2/footstand.py
src/unilab/training/rsl_rl.py
src/unilab/assets/robots/go2/go2.xml
```
