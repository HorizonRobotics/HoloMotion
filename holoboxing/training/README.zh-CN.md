# 拳击高层策略训练

[English](README.md) | 简体中文

训练高层拳击策略时，**HoloMotion V2 底层控制器保持冻结**。
高层策略预测 5 个身体点的目标位姿，HoloMotion 将其转换为关节动作。
先用参考动作进行监督学习，再在比赛擂台内与人形沙袋对练。

**示例说明**：本目录的训练方法、配置和命令均为示例（sample）。
用于熟悉训练、导出和评测流程，或作为自行开发的参考。

训练示例不定义比赛裁判。
导出策略后，须按当前 [v10 比赛规则](../evaluation/README.zh-CN.md#比赛规则)评测。
观测语义见[观测说明](../evaluation/README.zh-CN.md#观测)。
目标预测范围仍为未来 20–100 ms。
该范围与裁判的 20 ms 击打窗口含义不同。

```text
准备数据 -> 监督训练 -> 强化学习 -> 导出模型 -> 评测
```

## 环境与命令

先准备 HoloMotion [训练环境](../../docs/environment_setup.md)和 NVIDIA GPU。
为拳击训练单独创建环境。
随后在仓库根目录执行：

```bash
conda create --name holoboxing --clone holomotion_train
conda activate holoboxing
export Train_CONDA_PREFIX="$CONDA_PREFIX"
"$Train_CONDA_PREFIX/bin/python" -m pip install -r holoboxing/training/requirements.txt
```

## 准备资源与数据

下载[比赛资源](../evaluation/README.zh-CN.md#源码使用所需资源)，使 `holoboxing/assets` 可用。
下载冻结的底层控制器：

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing download --directory models/holomotion-v2
```

按[数据准备文档](../data/README.zh-CN.md)配置所需环境。
使用同一套比赛资源处理示例动作。
以下示例使用合并数据集的输出：`holoboxing/data/processed/boxing/prepared/`。
若仅处理 RoboStriker 或 InterHuman，请替换输入路径。
将 `--data` 和 `--amp-motions` 指向对应的 `prepared/` 目录。

| 输入 | 内容 | 参数 |
| --- | --- | --- |
| `holoboxing/assets` | 比赛机器人和擂台资源 | `--assets` |
| `models/holomotion-v2` | 冻结的 HoloMotion 控制器 | `--lower` |
| `holoboxing/data/processed/boxing/prepared` | 已准备的参考动作 | `--data` 或 `--amp-motions` |

传入目录，而非单个模型文件或清单文件。
每套资源或模型的配套文件须保持在一起。

同一份准备好的数据用于两个阶段：

- **监督学习：**参考状态作为输入，未来参考位姿作为目标。
- **AMP（Adversarial Motion Prior，对抗式动作先验）：**在强化学习中，参考动作与拳击奖励共同引导策略，使动作更自然。

## 监督训练

```bash
CUDA_VISIBLE_DEVICES=0 "$Train_CONDA_PREFIX/bin/python" -m holoboxing.training.train_sl \
  --assets holoboxing/assets \
  --lower models/holomotion-v2 \
  --data holoboxing/data/processed/boxing/prepared \
  --output results/boxing-sl \
  --epochs 8 \
  --device cuda:0
```

`--epochs` 指定完整遍历数据的次数。
此示例保存 `results/boxing-sl/epoch_000008.pt`，供下一阶段使用。

## 强化学习

从监督训练模型初始化，与人形沙袋对练：

```bash
CUDA_VISIBLE_DEVICES=0 "$Train_CONDA_PREFIX/bin/python" -m holoboxing.training.train_rl \
  --assets holoboxing/assets \
  --lower models/holomotion-v2 \
  --amp-motions holoboxing/data/processed/boxing/prepared \
  --supervised-checkpoint results/boxing-sl/epoch_000008.pt \
  --output results/boxing-rl \
  --updates 40000 \
  --device cuda:0
```

`--updates` 指定 PPO（近端策略优化）迭代次数。

## 导出与评测

将训练好的进攻策略导出为单个 ONNX 文件：

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.training.export \
  --checkpoint results/boxing-rl/model_40000.pt \
  --output results/boxing.onnx
```

提交导出的 `.onnx` 文件，不要提交训练用的 `.pt` 权重。
按[评测文档](../evaluation/README.zh-CN.md#运行)验证模型、运行对战并查看回放。
配置好评测镜像后执行：

```bash
./holoboxing/run.sh --validate results/boxing.onnx
```

## 修改策略

| 修改内容 | 代码入口 |
| --- | --- |
| 策略网络 | [rsl_rl/modules/actor_critic.py](rsl_rl/modules/actor_critic.py) |
| 观测 | [envs/mujoco_env.py](envs/mujoco_env.py) 和 [rsl_rl/modules/normalization.py](rsl_rl/modules/normalization.py) |
| 拳击及沙袋奖励 | [tasks/combat/mdp/rewards.py](tasks/combat/mdp/rewards.py) 和 [envs/mujoco_rewards.py](envs/mujoco_rewards.py) |
| PPO、AMP 和环境配置 | [configs/training.json](configs/training.json) |

修改后仍须符合[比赛策略接口](../evaluation/README.zh-CN.md#策略接口)：
输入为 `obs[1,1097]`，输出为 `target_pose[1,225]`，底层控制器保持冻结。

## 训练方法与来源

本示例用于说明开发流程，不代表复现了所引论文的效果。

| 工作 | 沿用内容 | 本示例的调整 |
| --- | --- | --- |
| Yin 等，[RoboStriker: Latent-Space Strategic Games for Autonomous Humanoid Boxing](https://arxiv.org/abs/2608.16195)，2026；[预览代码](https://github.com/yinkangning0124/RoboStriker-Preview) | 改编其沙袋训练结构、奖励项及 PPO/AMP 实现结构 | 高层任务网络、先验网络和解码器联合训练，输出五点目标位姿，再由冻结的 HoloMotion V2 执行 |
| [HoloMotion](https://github.com/HorizonRobotics/HoloMotion) | 直接使用冻结的 V2 跟踪控制器，并复用观测及归一化工具 | 只训练高层拳击策略，不重新训练底层控制器 |
| Schulman 等，[PPO](https://arxiv.org/abs/1707.06347)，2017；[广义优势估计（GAE）](https://arxiv.org/abs/1506.02438)，2016 | 使用裁剪策略更新目标和优势估计方法 | 本地实现处理拳击双方角色，以及高层目标动作 |
| Peng 等，[AMP：对抗式动作先验](https://xbpeng.github.io/projects/AMP/)，2021 | 用动作判别器提供额外的动作风格奖励 | 使用 G1 状态转移特征及 RoboStriker 风格的二元交叉熵训练，并结合拳击任务奖励 |
| Schwarke 等，[RSL-RL: A Learning Library for Robotics Research](https://arxiv.org/abs/2509.10771)，2025；[代码](https://github.com/leggedrobotics/rsl_rl) | 训练模块沿用其实现脉络和版权声明 | 随附的双角色 PPO/AMP 模块已作适配，并非原封不动的上游安装包；见[许可](rsl_rl/LICENSE) |
| Luo 等，[SMPLOlympics: Sports Environments for Physically Simulated Humanoids](https://arxiv.org/abs/2407.00187)，2024 | 沿用源码注明的判别器初始化和梯度惩罚约定 | 训练对象为比赛 G1 机器人，采用本任务的特征和奖励 |

两者都使用潜表示。
本示例的任务分支与历史先验分支生成归一化的 32 维潜变量。
联合训练的解码器结合该潜变量和当前状态，输出 `target_pose[1,225]`。
PPO 在这 225 维目标上采样并计算动作概率。
内部潜变量并非当前 PPO 的随机动作空间。
冻结的 HoloMotion V2 再将目标转换为关节动作。
RoboStriker 的策略则为预训练动作解码器选择潜变量动作。
内部采用低维表示，不能单独证明训练或推理加速。
加速效果仍需通过对照测量确认。
上文命令仅覆盖监督初始化和沙袋训练。
不包含论文完整的潜空间虚拟自博弈流程。
这些示例也未建立与 RoboStriker 的效果对比。

实现入口：[策略网络](rsl_rl/modules/actor_critic.py)、[PPO](rsl_rl/algorithms/ppo.py)、[优势估计](rsl_rl/storage/rollout_storage.py)、[AMP](rsl_rl/modules/amp.py)和[判别器](rsl_rl/modules/amp_discriminator.py)。
HoloMotion 的研究引用为 Chen 等的 [HoloMotion-1 Technical Report](https://arxiv.org/abs/2605.15336)，2026。
该报告介绍 V1；V2 的行为及接口以当前代码和模型发布为准。
