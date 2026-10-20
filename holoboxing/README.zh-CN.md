# HoloBoxing

[English](README.md) | 简体中文

**参赛须遵守评测接口与规则。数据与训练示例均为可选参考，选手可以自行开发策略。**

| 内容 | 对选手的要求或用途 |
| --- | --- |
| 必须遵守 | 提交符合策略接口、文件限制及官方硬件与耗时要求的 ONNX 模型；对战按比赛规则计分 |
| 本地评测 | 使用与云端相同的评测镜像，验证模型、运行对战并查看回放；建议在提交前使用 |
| 可选参考 | `data/` 和 `training/` 演示开发流程；选手可以自行选择数据、网络结构和训练方法 |

ONNX 是提交模型使用的通用神经网络格式。
已有兼容的 ONNX 策略可直接评测，无需先运行数据或训练示例。

## 本地评测

请使用本轮比赛指定的评测镜像。
每次结果须记录已核验的镜像 ID 或仓库摘要。
仅标签相同，不能证明与云端版本一致。

模型验证和对战需要 Linux、Docker、NVIDIA GPU，以及 NVIDIA Container Toolkit。
回放只需要 Docker 和浏览器。
镜像已包含控制器、机器人资源、擂台和播放器。
无需另行下载这些资源，也无需在主机配置 Python 环境。
当前版本为 **v10**。
该版本统一观测时序，并更新击打规则。
收到 v10 归档后，先核验随附校验值，再导入。

```bash
sha256sum --check SHA256SUMS
docker load --input holoboxing-eval-v10-amd64.tar.gz
export HOLOBOXING_IMAGE="horizonrobotics/holomotion:holoboxing-eval-v10-amd64"
docker image inspect --format '{{.Id}}' "$HOLOBOXING_IMAGE"
```

摘要固定、镜像导入及主机要求见[运行说明](evaluation/README.zh-CN.md#运行)。
上述 v10 标签用于选择导入的归档。
本文未声明该版本已发布至 Docker Hub。
源码开发使用当前检出版本固定的[已发布资源](evaluation/README.zh-CN.md#源码使用所需资源)。

`my_policy.onnx` 是自己的模型，`opponent_policy.onnx` 是用于对战的对手模型。
请将示例路径替换为实际文件路径。
第一个模型控制红方，第二个控制蓝方。
在 HoloMotion 仓库根目录执行：

```bash
./holoboxing/run.sh --validate models/my_policy.onnx
./holoboxing/run.sh --match models/my_policy.onnx models/opponent_policy.onnx
./holoboxing/run.sh --view results/match/replay.hdf5
```

验证结果写入 `results/validation.json`。
对战得分和事件写入 `results/match/match.json`。
同目录的 `replay.hdf5` 可在浏览器中回放。
已有结果不会被覆盖，可用 `--output` 指定其他输出路径。
本地命令不会提交模型，也不会更新比赛排行榜。

## 提交要求

提交一个完整的高层拳击策略 ONNX 模型。
冻结的 HoloMotion V2 底层控制器由评测环境提供。

| 项目 | 要求 |
| --- | --- |
| 模型文件 | 单个 `.onnx` 文件，严格小于 50,000,000 字节；不支持外部张量文件 |
| 输入 | `obs`，float32，形状为 `[1,1097]`；包含自身状态历史、对手位置、剩余时间和双方分差 |
| 输出 | `target_pose`，float32，形状为 `[1,225]`；对应 5 个未来时刻 × 5 个身体点 × 9 个位姿分量 |
| 控制频率 | 50 Hz；每次输出覆盖未来 20–100 ms 的目标位姿 |
| 初赛耗时要求 | 在 RTX 5090 上，策略与底层控制器的完整推理链平均耗时 ≤ 5 ms |

观测索引、身体点顺序、坐标系、旋转表示和导出限制见[策略接口](evaluation/README.zh-CN.md#策略接口)。
计时条件和官方硬件检查见[验证与排错](evaluation/README.zh-CN.md#验证与排错)。

## 计分与胜负

每回合最多持续 60 秒仿真时间。
手部接触对手头部或躯干，满足条件后得 1 分。
击倒判定成立后得 1,000 分，回合立即结束。
若未分出击倒胜负，则在时间结束时比较得分。
得分高者获胜，同分为平局。

v10 的法向冲量阈值为 **5 N·s**。
累计窗口为 **20 ms**，每只手冷却 **200 ms**。
得分会消耗累计冲量，无需先脱离接触。

接触阈值、得分冷却时间、倒地标准和同时击倒的处理方式见[比赛规则](evaluation/README.zh-CN.md#比赛规则)。

## 为什么使用 HoloMotion

比赛策略接口将拳击决策与全身动作执行分开：

- **用五点目标控制全身。** 高层策略指定骨盆、双腕和双踝的目标位姿。
  HoloMotion 将目标转换为 29 个关节的协调动作。
  选手可围绕出拳、防守和移动设计身体目标，无需逐一编排关节动作。
- **复用运动预训练能力。** 预训练控制器提供全身动作跟踪能力。
  底层保持冻结，选手可专注开发高层拳击策略。
  无需从头训练动作跟踪控制器。

五点接口简化的是任务表达。
每次输出仍包含 5 个未来时刻的目标位姿，共 225 个值。
底层能力和模型版本见 [HoloMotion V2 说明](../README.md#holomotion-v2-flexible-sparse-to-whole-body-command-following-with-cross-modal-extensibility)及[固定控制器配置](evaluation/frozen-model.json)。
实际拳击效果取决于选手的策略及其生成的目标。

## 可选的数据与训练示例

为降低开发门槛，我们提供数据准备和训练示例。
它们演示数据准备、监督初始化、强化学习、ONNX 导出和评测流程。

这些内容供开发参考，并非参赛必经步骤。
选手可自行准备数据，也可采用自己的训练实现。
导出的策略符合评测要求即可。
示例不以达到领域最佳水平（SOTA）为目标。
如需沿用参考流程，可从[示例数据文档](data/README.zh-CN.md)和[示例训练文档](training/README.zh-CN.md)开始。
源码使用所需资源见[资源准备](evaluation/README.zh-CN.md#源码使用所需资源)。

### 示例方法与来源

参考训练流程改编自 [RoboStriker](https://arxiv.org/abs/2608.16195) 的沙袋训练结构及部分实现。
两者都使用潜表示。
本示例将内部 32 维潜变量解码为 225 维五点目标位姿。
目标由冻结的 HoloMotion V2 执行。
当前 PPO 在目标位姿上采样；RoboStriker 的策略动作则位于潜空间。
选手无需采用本示例的网络结构。

复用内容和具体调整分别见[训练方法与来源](training/README.zh-CN.md#训练方法与来源)、[数据与方法来源](data/README.zh-CN.md#数据与方法来源)、[研究与资源来源](assets/smpl-ring-v2/SOURCE.md)，以及[旋转表示约定](evaluation/README.zh-CN.md#目标位姿)。

## 目录说明

| 目录 | 主要用途 |
| --- | --- |
| [evaluation/](evaluation/README.zh-CN.md) | 通用评测接口、比赛规则、模型验证、对战和回放 |
| [assets/](assets/README.zh-CN.md) | 机器人、擂台和回放资源；Docker 评测镜像已内置 |
| [data/](data/README.zh-CN.md) | 可选的示例数据准备流程 |
| [training/](training/README.zh-CN.md) | 可选的示例策略训练和 ONNX 导出流程 |
