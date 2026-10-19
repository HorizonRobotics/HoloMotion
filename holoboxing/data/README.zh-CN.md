# 拳击示例数据

[English](README.md) | 简体中文

将 RoboStriker 和 InterHuman 动作转换为训练输入。
产物供拳击训练器和 HoloMotion V2 使用。
所有命令均在 HoloMotion 仓库根目录执行。

**示例说明**：数据与处理流程均为示例（sample）。
用于熟悉流程，或作为自行开发的参考。

| 数据源 | 处理流程 |
| --- | --- |
| RoboStriker | 下载已重定向的 G1 机器人 NPZ，直接执行 `prepare` |
| InterHuman | 下载双人 SMPL 动作，执行 `retarget`，再执行 `prepare` |

可单独使用任一数据源，也可合并使用。
RoboStriker 已是机器人动作，无需再次进行 SMPL 重定向。

## 环境与资源

使用 [HoloMotion 训练环境](../../train.env)和[比赛资源](../evaluation/README.zh-CN.md#源码使用所需资源)。
InterHuman 重定向需要支持 CUDA 的 GPU，以及已取得许可的中性 SMPL 人体模型。
请通过 [SMPL 官方下载入口](https://smpl.is.tue.mpg.de/)获取该模型。
本仓库和动作数据集均不包含此模型。

```bash
source train.env
"$Train_CONDA_PREFIX/bin/python" -m pip install -r holoboxing/data/requirements.txt
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data --help
```

下载和生成文件应放入被 Git 忽略的数据目录。
数据准备和训练须使用同一套比赛资源。

## 下载并转换 RoboStriker

下载 [RoboStriker-Preview](https://github.com/yinkangning0124/RoboStriker-Preview) 发布的 G1 动作。
仓库内的[源文件清单](profiles/robostriker.json)固定上游版本。
清单还记录下载标识、文件大小和 SHA256 校验值。

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data download robostriker \
  --output holoboxing/data/raw/RoboStriker --workers 8

"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data prepare \
  --source holoboxing/data/raw/RoboStriker \
  --output holoboxing/data/processed/RoboStriker \
  --assets holoboxing/assets --workers 2

"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data verify \
  --output holoboxing/data/processed/RoboStriker --assets holoboxing/assets
```

`prepare` 直接读取下载的 G1 状态。
它按名称排列关节，并重采样至 50 Hz。
随后使用比赛机器人的运动学生成训练数据。
该过程不会再次进行人体到机器人的重定向。
小规模下载检查可加 `--limit-groups 1`。
每组包含原始动作及其镜像版本。
`download.json` 记录来源标识和文件校验值。
再次运行会校验缓存，并续传中断文件。
可用 `--repair` 替换损坏的缓存文件。

## 下载 InterHuman 示例

示例使用仓库内的[片段清单](profiles/interhuman_training_files.json)，共 238 个片段。
其中 220 个属于官方训练集，18 个属于验证集。
不包含官方测试片段。从公开来源下载：

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data download interhuman \
  --output holoboxing/data/raw/InterHuman --workers 8
```

默认命令按固定清单下载，不搜索标注文本。
小规模检查可加 `--limit-clips 2`。
如需其他源片段，可重复传入 `--clip-id ID` 替换默认清单。
`download.json` 记录请求的 ID 和下载文件哈希。
使用内置清单时，也记录其哈希。
已有文件会先校验；下载中断后再次运行可续传。
可用 `--repair` 替换损坏的缓存文件。

[InterHuman](https://github.com/tr3e/InterGen#interhuman-dataset) 仅允许非商业使用，禁止再分发数据集。
请从官方来源下载原始文件，并保留 `LICENSE.md`。
不得随代码或数据再分发受许可约束的 SMPL 模型。

## 重定向 InterHuman 动作

将下载的双人 SMPL 动作转换为 G1 动作：

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data retarget \
  --source holoboxing/data/raw/InterHuman \
  --output holoboxing/data/converted/InterHuman \
  --smpl-model /path/to/licensed/neutral-smpl.pkl \
  --assets holoboxing/assets --device cuda:0 --iterations 4
```

源目录须包含示例下载记录 `download.json`。
重定向仅处理该记录中的片段，并保留官方训练、验证和测试划分。
本例执行 4 次 HoloRetarget 迭代，不改变其全局默认值。

`motions.json` 记录源文件及模型哈希、资源标识和转换配置。
输出为按名称排列的 G1 29 关节状态，帧率为 50 Hz。

## 将 InterHuman 转换为训练输入

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data prepare \
  --source holoboxing/data/converted/InterHuman \
  --output holoboxing/data/processed/InterHuman \
  --assets holoboxing/assets --workers 2

"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data verify \
  --output holoboxing/data/processed/InterHuman --assets holoboxing/assets
```

## 合并两个数据源

如需合并数据，在同一命令中重复传入 `--source`：

```bash
"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data prepare \
  --source holoboxing/data/raw/RoboStriker \
  --source holoboxing/data/converted/InterHuman \
  --output holoboxing/data/processed/boxing \
  --assets holoboxing/assets --workers 2

"$Train_CONDA_PREFIX/bin/python" -m holoboxing.data verify \
  --output holoboxing/data/processed/boxing --assets holoboxing/assets
```

输入须为 RoboStriker 下载目录和 InterHuman 重定向目录。
不要传入此前生成的 `prepared/` 目录。
RoboStriker 的原始动作与镜像动作归入同一划分。
没有官方划分时，默认种子为 42。
验证集比例为 0.1，按原始动作组分配。
InterHuman 保留官方划分，并排除官方测试片段。

## 产物与检查

上述 `prepare` 命令生成相同的目录结构。
输出须使用源数据目录之外的新目录。
包含 `FAILED.json` 的输出不能视为完整数据集。

| 输出 | 用途 |
| --- | --- |
| `prepared/` | 当前拳击训练器使用的观测、未来位姿目标和动作先验数据 |
| `robot_h5/train/` | HoloMotion V2 训练参考动作 |
| `robot_h5/validation/` | HoloMotion V2 验证参考动作 |
| `sources.json`、`COMPLETE.json` | 数据来源、划分和校验结果 |

将所选数据集的 `prepared/` 目录传给[示例训练流程](../training/README.zh-CN.md#环境与命令)。
RoboStriker 提供单机器人参考动作。
其拳击观测中的对手为合成的站立机器人。
InterHuman 则保留双方的录制动作。
仅通过数据转换和读取检查，不能证明物理跟踪或拳击效果。

## 数据与方法来源

| 来源 | 沿用内容 | 本示例的使用范围 |
| --- | --- | --- |
| Liang 等，[InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions](https://arxiv.org/abs/2304.05684)，2024 | 使用该工作发布的 InterHuman 双人动作数据集 | 将指定的录制片段重定向到 G1，不使用 InterGen 的文本条件扩散生成模型 |
| Loper 等，[SMPL: A Skinned Multi-Person Linear Model](https://smpl.is.tue.mpg.de/)，2015 | 使用已获许可的人体模型，提供体型及骨架信息 | SMPL 描述源人体；转换结果使用比赛 G1 机器人及其关节定义 |
| [HoloRetarget](../../docs/motion_retargeting.md) | 直接使用现有的人体到机器人重定向实现 | 本适配器处理双人动作、统一水平缩放和比赛机器人资源 |
| [RoboStriker-Preview](https://github.com/yinkangning0124/RoboStriker-Preview) | 使用公开 G1 动作 NPZ，包含镜像动作 | 将机器人状态转换为比赛训练数据和 HoloMotion V2 参考动作，无需再次进行 SMPL 重定向 |

HoloRetarget 属于 HoloMotion 项目，实现说明见上方文档。
项目研究引用为 Chen 等的 [HoloMotion-1 Technical Report](https://arxiv.org/abs/2605.15336)，2026。
该报告介绍 V1；V2 控制接口以当前仓库文档为准。
