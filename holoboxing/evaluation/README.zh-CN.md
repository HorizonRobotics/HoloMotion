# HoloBoxing 评测

[English](README.md) | 简体中文

验证高层拳击策略，使用冻结的 HoloMotion V2 控制器运行双机器人对战，并回放比赛结果。
回放只读取已保存的状态和事件，不运行策略推理或物理步进。

| 组件 | 主要功能 |
| --- | --- |
| `cli.py`、`validation.py`、`qualification.py`、`benchmark.py` | 命令行入口、模型接口检查、硬件达标检查和完整推理链计时 |
| `match.py`、`runtime/` | 高层策略与冻结控制器执行、GPU 物理仿真、击打与倒地判定、计分及对战诊断 |
| `export_hdf5.py`、`replay_schema/`、`viewer/`、`viewer_server.py` | 保存回放数据、启动浏览器播放器，并导出带事件录音的视频 |
| `resources.py`、`models.py`、`frozen-model.json` | 校验资源清单和固定版本的底层控制器文件包 |
| `build.sh`、`scripts/`、`docker/` | 按允许列表准备构建上下文，构建包含评测依赖的镜像 |

在 HoloMotion 仓库根目录使用 `./holoboxing/run.sh`。
该脚本通过 Docker 完成验证、对战和回放。
Python 入口为 `python -m holoboxing`。
训练命令见 [`training/`](../training/README.zh-CN.md)。
数据和训练文档提供示例（sample），用于熟悉流程或作为开发参考。

## 文档导航

- [运行](#运行)
- [比赛规则](#比赛规则)
- [策略接口](#策略接口)
- [验证与排错](#验证与排错)
- [构建镜像](#构建镜像)

训练或从源码评测时，先准备[源码使用所需资源](#源码使用所需资源)。

**请使用本轮比赛指定的评测镜像。**
须核验归档校验值和镜像 ID，或已发布的仓库摘要。
与云端核对后，才能声明版本一致。
仅标签相同，不能保证镜像内容相同。

镜像已包含 **HoloMotion V2 模型权重、机器人 URDF、网格模型、擂台纹理和浏览器播放器**。
这些资源无需另行下载。
选手只需提供高层拳击策略的 ONNX 模型。
ONNX 是用于交换神经网络模型的通用文件格式。

## 运行

验证和对战需要 Linux、Docker、NVIDIA 驱动及 NVIDIA Container Toolkit。
Container Toolkit 使容器能够访问 GPU。
回放只需要 Docker 和浏览器。
使用镜像时，主机无需配置 Python 或 npm。

主机操作流程已在 **Ubuntu 22.04.4 LTS 和 Docker 27.1.2** 上测试。
评测镜像内部使用 Debian 12（Bookworm）。
官方评测要求 **RTX 5090**。
其他主机操作系统尚未在此验证。

当前版本为 **v10**。
策略位姿和速度来自同一当前状态。
冲量窗口为 20 ms，阈值为 5 N·s。
每只手的计分间隔为 200 ms。
此前的 v9 仍保留原规则和观测行为。

收到 Linux amd64 的 v10 归档后，先核验再导入。

```bash
sha256sum --check SHA256SUMS
docker load --input holoboxing-eval-v10-amd64.tar.gz
export HOLOBOXING_IMAGE="horizonrobotics/holomotion:holoboxing-eval-v10-amd64"
docker image inspect --format '{{.Id}}' "$HOLOBOXING_IMAGE"
```

最后一条命令显示镜像 ID。
请与随附构建记录核对。
将已核验的 ID 赋给 `HOLOBOXING_IMAGE`，可固定本地版本。
上述 v10 标签用于选择导入的归档。
本文未声明该版本已发布至 Docker Hub。
若比赛提供仓库镜像，请拉取其确切摘要。
引用格式为 `horizonrobotics/holomotion@sha256:...`。
本地镜像 ID 与仓库摘要标识的对象不同。
源码开发可按[构建说明](#构建镜像)制作镜像。

`./holoboxing/run.sh --init` 仍为预留入口，不执行安装或下载。
请显式导入或拉取所选镜像。
随后在 HoloMotion 仓库根目录执行：

`my_policy.onnx` 为自己的模型，`opponent_policy.onnx` 为对手模型。
请替换为实际文件路径。
第一个模型控制红方，第二个控制蓝方。

```bash
./holoboxing/run.sh --validate models/my_policy.onnx
./holoboxing/run.sh --match models/my_policy.onnx models/opponent_policy.onnx
./holoboxing/run.sh --view results/match/replay.hdf5
```

若收到镜像归档文件，使用 `docker load --input image.tar.gz` 导入。
然后选择已核验的标签或镜像 ID。
源码构建适用于开发，不能据此认定与云端固定镜像一致。

| 操作 | 输出 |
| --- | --- |
| `--validate MODEL` | `results/validation.json`：模型检查、完整推理链实测耗时和硬件达标检查 |
| `--match YOUR_POLICY OPPONENT_POLICY` | `results/match/match.json`：得分、胜者、事件和诊断；`replay.hdf5`：录制状态 |
| `--view REPLAY` | 在 `http://localhost:8080` 打开播放器，支持播放、暂停、跳转、事件标记和 MP4 导出 |

已有结果不会被覆盖。
使用 `--output` 更换结果路径，`--gpu` 选择主机 GPU，或 `--port` 更换播放器端口。
全部选项见 `./holoboxing/run.sh --help`。
模型以只读方式挂载，容器使用当前用户身份运行。
`--assets` 和 `--lower` 可替换内置资源及控制器。
正常使用无需指定这两个参数。

播放器读取已保存的状态，不执行策略或物理步进。
录制的 `HIT` 和 `KO` 事件分别触发选定的沙袋击打录音和女声 K.O. 音效。
两份录音均采用 CC0 授权。
来源与处理说明位于播放器的 `notices/recorded-effects.txt`。
跳转进度时不播放音效。
声音开关同时控制回放和 MP4 导出。
导出会保留结束语音的尾音。

导出 MP4 时，请保持浏览器标签页可见。
按 Ctrl+C 停止服务器。
MP4 编码需要兼容的 Chrome / Edge 浏览器，并通过 HTTPS 或 localhost 访问。
使用远程服务器时，可通过 SSH 将播放器端口转发到本地。
也可使用浏览器信任的证书运行：
`python -m holoboxing view replay.hdf5 --host 0.0.0.0 --tls-cert cert.pem --tls-key key.pem`。
私钥须保存在仓库之外。

## 比赛规则

每回合最多持续 **60 秒仿真时间**。
控制频率为 50 Hz，即每 20 ms 输出一次目标。
物理仿真频率为 200 Hz，即每 5 ms 推进一步。
仿真时长不同于命令实际运行耗时。

| 判定 | 规则 |
| --- | --- |
| 有效击打 | 手部接触对手头部或躯干。**20 ms** 内法向冲量累计达 **5 N·s**，得 **1 分**。 |
| 连续得分 | 每只手的得分冷却时间为 **200 ms**。得分会消耗累计冲量，无需先脱离接触。 |
| 倒地 | 骨盆或躯干的刚体原点距擂台地面的高度严格小于 **0.30 m**；仅倾斜不算倒地 |
| 击倒（KO） | 连续倒地 **3 s** 后确认击倒；对手得 **1,000 分**，回合结束 |
| 胜负 | 单方确认击倒时，对手获胜；双方同时确认倒地时为平局；超时则得分高者获胜，同分为平局 |

法向冲量是接触面法向力对时间的积分，单位为牛顿秒（N·s）。
摩擦力、峰值力和接近速度不能替代该阈值。
可结合录制事件与接触诊断分析得分原因。
结果记录计分版本和具体配置数值。
倒地先后按 50 Hz 观测时间戳判定。
时间戳相同即视为同时发生。
不会用事件数组的存储顺序决定胜负。

## 策略接口

**高层拳击策略**预测 5 个身体点的未来位姿。
镜像内置的 **HoloMotion 冻结底层控制器**将目标转换为关节动作。
只提交高层策略即可。
模型必须为自包含文件，严格小于 **50,000,000 字节**。
不支持外部张量文件。

张量可以理解为多维数组。
运行时使用批大小 1，数据类型为 float32，不自动归一化。
批维度可以是符号维度，但须支持大小 1。

| 方向 | 张量名称 | 类型 | 形状 |
| --- | --- | --- | --- |
| 输入 | `obs` | float32 | `[1,1097]` |
| 输出 | `target_pose` | float32 | `[1,225]` |

ONNX 元数据 `policy_contract` 为可选项。
如提供，其值必须为 `holomotion_v2_boxing_obs1097_target_pose225_v1`。

请在推理模式下导出模型。
不支持随机 ONNX 算子和训练模式的 Dropout。
分支与局部函数内的算子同样受此限制。
对较宽维度执行 TopK 选择时，要求最后一维宽度已知。
`K` 必须为 1 至 128 之间的常量。
不支持的选择方式会导致验证失败。

### 坐标系

“自身机器人”指运行该策略的机器人，“对手”指另一台机器人。
以下坐标定义均以自身机器人为参照。

| 坐标系 | 原点 | 轴朝向 |
| --- | --- | --- |
| 自身骨盆坐标系 | 当前实测骨盆位置 | 跟随骨盆的完整三维旋转，包含横滚、俯仰和偏航 |
| 自身航向坐标系 | 当前实测骨盆位置 | 仅跟随骨盆偏航角；Z 轴始终与世界坐标系竖直方向一致 |

### 观测

索引从 0 开始，区间包含起点、不包含终点。
展平时，最后一维的索引变化最快。

| `obs` 切片 | 含义 | 展平前形状 |
| --- | --- | --- |
| `[0:1080]` | 从最早到当前的 10 帧自身实测状态，间隔 20 ms | `[10,108]` |
| `[1080:1095]` | 对手的 5 个身体点位置，位于自身当前骨盆坐标系中 | `[5,3]` |
| `[1095:1096]` | 本回合剩余时间，单位为秒 | `[1]` |
| `[1096:1097]` | 自身得分减去对手得分 | `[1]` |

每帧自身状态包含 108 个值：

| 帧内切片 | 含义 | 形状 |
| --- | --- | --- |
| `[0:3]` | 骨盆角速度，使用采样时刻的机体坐标系，单位为弧度/秒 | `[3]` |
| `[3:48]` | 5 个实测身体点位姿，位于自身当前航向坐标系中 | `[5,9]` |
| `[48:63]` | 身体点线速度，使用自身当前航向坐标轴，单位为米/秒 | `[5,3]` |
| `[63:108]` | 前一采样时刻的身体点位姿，位于自身当前航向坐标系中 | `[5,9]` |

历史位姿统一转换到自身**当前**航向坐标系。
这里既使用当前朝向，也使用当前原点。
线速度只按当前航向坐标轴转换，不减去骨盆速度。
角速度保留采样时刻的机体坐标系。
当前帧位于 `obs[:,972:1080]`。
重置后，历史序列各帧均重复第一帧实测状态。
该帧的前一时刻位姿设为其当前位姿。
每步先生成观测，再执行该步动作。

每帧位姿、线速度和骨盆角速度使用同一当前状态。
该状态由关节位置 `qpos` 和速度 `qvel` 确定。
观测刷新独立执行，保留物理求解器及接触状态。

### 目标位姿

将 `target_pose` 重排为 `[1,5,5,9]`。
各维依次为批次、未来时刻、身体点、位姿分量。

| 维度 | 顺序 |
| --- | --- |
| 未来时刻 | 当前观测后的 20、40、60、80、100 ms |
| 身体点 | 骨盆、左腕、右腕、左踝、右踝 |
| 精确刚体名称 | `pelvis`、`left_wrist_yaw_link`、`right_wrist_yaw_link`、`left_ankle_roll_link`、`right_ankle_roll_link` |
| 位姿分量 | `px,py,pz,Xx,Xy,Xz,Zx,Zy,Zz` |

每个身体点使用 **3 维位置 + 6 维旋转表示**，共 9 个值。
位置单位为米。
旋转采用 Zhou 等人在以下论文中提出的连续 6 维表示：
[*On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019](https://arxiv.org/abs/1812.07035)。

本接口保存旋转矩阵的**第一列和第三列**：
`X = (Xx,Xy,Xz)`，`Z = (Zx,Zy,Zz)`。
第二列由 `Y = Z × X` 恢复，得到 `R = [X Y Z]`。
模型输出的 X/Z 轴必须互相正交，且均为单位向量。
评测器以 `1e-3` 为容差进行检查。

Zhou 等的论文提供了连续 6 维旋转表示方法。
上面的 X/Z 列顺序是本接口的具体约定。
不能直接套用其他实现中的 6 维分量顺序。

所有未来目标均在自身当前骨盆坐标系中表示。

输出位置中的非有限值会替换为零。
若某个朝向分量为非有限值，该身体点的 X/Z 轴会替换为单位旋转对应的轴。
诊断会记录修正次数。
有限值构成的非法朝向仍会报错。
实测状态、物理状态、冻结模型特征和循环状态缓存均不做修正。
底层控制器接收全部 5 组未来身体点目标。
其数据组织、有效性掩码、权重和状态保持固定。

## 验证与排错

本地验证检查模型在当前 GPU 上的兼容性。
这不能证明拳击水平，也不能替代官方硬件验收。
固定站立策略以 0:0 结束属于正常结果。
推理和物理仿真均不提供 CPU 回退。

初赛硬件要求为 **RTX 5090**。
预热 100 步后，测量 1,000 步。
完整推理链平均耗时须不超过 **5 ms**。
计时包含高层观测、高层推理、解码、底层观测、底层推理、缓存更新和目标转换。
输出修正与物理仿真单独计时。
返回的循环状态会复用，并按冻结控制器的上下文长度重置。
观测计时包含独立执行的正向运动学更新。

报告记录实际软硬件、模型哈希、耗时分位数和结果一致性检查。
`--qualification official` 显式检查官方硬件要求。
其他 GPU 会报告 `PLATFORM_WRONG_TARGET_GPU`。
其他显卡的测量值不能按固定倍数换算。
本地对战不会更新比赛排行榜。

如需检查完整控制帧的时限，单独运行耗时预算压力测试：

```bash
python -m holoboxing.evaluation.benchmark my_policy.onnx opponent_policy.onnx \
  --assets /path/to/assets --lower /path/to/lower \
  --output-dir /path/to/new-benchmark --policy-budget-ms 10
```

程序先逐一检查完整的高层策略与 HoloMotion v2.0 推理链。
每条链仍须满足平均 5 ms 的要求。
随后运行完整对战，为每个测量帧中的两条策略链合计预留 10 ms。
真实模型和状态转换仍会执行。
预留时通过等待补足耗时，并计入超出预定时长的调度开销。
这种方式不能模拟任意 10 ms 策略的真实 GPU 负载。
输出修正、物理仿真、裁判计算和内存中的回放数据准备另占帧耗时。
冷启动、最终文件编码和浏览器回放不计入稳态范围。

预热 100 帧后，完整帧耗时的 **P99 必须严格小于 20 ms**，否则命令失败。
P99 指耗时的第 99 百分位数。
计算使用相邻的较高观测值（`higher`），不会通过插值掩盖超时。
报告保留模型哈希、实际 GPU、驱动、负载、单链测量结果及每次预留样本。
请在目标 GPU 上重跑检查。
重复完整对战，以收集尾部耗时证据。

| 现象或错误 | 处理方式 |
| --- | --- |
| `Image unavailable` | 导入提供的镜像，检查 `--image` 或 `HOLOBOXING_IMAGE` |
| `File does not exist` | 检查工作目录和输入路径 |
| `Output already exists` | 用 `--output` 指定新路径 |
| CUDA/GPU 不可用 | 检查驱动、Container Toolkit 和 Docker GPU 访问权限 |
| `MODEL_TOO_LARGE` | 确保自包含模型文件严格小于大小上限 |
| `FORMAT_CHECK_FAILED` | 导出有效 ONNX 文件，不使用外部张量存储 |
| `CONTRACT_CHECK_FAILED` | 检查张量名称、形状、类型及可选接口元数据 |
| `PLATFORM_WRONG_TARGET_GPU` | 本地兼容性检查使用默认 local 模式；官方硬件验收须使用指定 GPU |
| 机器人倒地 | 按策略接口和倒地规则，检查预测目标与录制状态 |

## 源码使用所需资源

已发布的拳击资源位于
[HorizonRobotics/HoloMotion_assets](https://huggingface.co/datasets/HorizonRobotics/HoloMotion_assets/tree/main/holoboxing)。
HoloMotion 已在 `thirdparties/HoloMotion_assets` 中跟踪该数据集。
`holoboxing/assets` 是指向其 `holoboxing/` 子目录的符号链接。
子模块版本由当前 HoloMotion 检出版本固定。
已发布的资源提交如下。

[`795e5e1c1d3b6b15cc1cea9e7752742715724826`](https://huggingface.co/datasets/HorizonRobotics/HoloMotion_assets/commit/795e5e1c1d3b6b15cc1cea9e7752742715724826)

该提交包含 `boxing-v6-resources`。
v9 与 v10 共用同一物理资源。
资源版本仍为 `boxing-v6-resources`。
资源版本独立于镜像及计分版本。
相同文件以新提交发布，无需因此重建镜像。
目录结构和来源说明见[资源文档](../assets/README.zh-CN.md)。

安装 Git LFS 后，在 HoloMotion 仓库根目录执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init thirdparties/HoloMotion_assets
git -C thirdparties/HoloMotion_assets lfs pull --include="holoboxing/**" --exclude=""
```

更新 HoloMotion 后，请重跑以上命令。
它们会取得代码固定的资源版本。
请保留子模块引用，不直接跟随资源仓库最新分支。
以上命令仅下载拳击资源，不拉取其他大型模型或数据文件。
仓库中的小文件和其他 LFS 指针文件也会被检出。
从源码评测、训练或构建镜像时，使用 `--assets holoboxing/assets`。
数据集仓库根目录不是拳击资源根目录。
冻结的 V2 控制器需另行下载。
Docker 评测镜像已包含资源和控制器。

在 HoloMotion 训练环境中校验下载资源：

```bash
source train.env
"$Train_CONDA_PREFIX/bin/python" -m holoboxing assets --assets holoboxing/assets
```

## 构建镜像

维护者需要支持 BuildKit 的 Docker，以及带完整清单的拳击资源目录。
还需准备 [frozen-model.json](frozen-model.json) 指定的冻结模型文件包。
从源码使用时，可通过 `python -m holoboxing download --directory /absolute/lower` 下载该包。
请从无未提交修改的 HoloMotion 仓库构建：

```bash
./holoboxing/evaluation/build.sh --assets /absolute/assets --lower /absolute/lower --context /absolute/new-context --image holoboxing-eval:local-v10
```

构建上下文须使用源码仓库之外的新目录。
脚本会固定公开基础镜像摘要，并校验资源和模型哈希。
可通过 `--help` 查看覆盖选项。
`--development` 会明确记录开发快照。
构建脚本不会发布镜像。
可设置 `HOLOBOXING_BUILD_PYTHON` 选择构建辅助程序的 Python 解释器。

戴拳套 URDF 是机器人参数源。
评测直接引用播放器使用的红蓝 XML。
擂台碰撞体读取播放器的 `smpl-ring-v2/scene.json`。
资源校验会拒绝旧机器人和裸手网格。
播放与评测的资源清单也须一致。
构建时会编译并比较两种格式的物理参数。
修改 URDF 后，须同步 XML 和清单：

```bash
python holoboxing/evaluation/scripts/sync_robot_assets.py --assets /absolute/assets --write
```

训练和评测共用双精度平面投影修复。
碰撞输入、输出及其余求解仍为单精度。
该精度修复保留原有接触和摩擦参数。
电机仍使用比例微分控制，即 PD 控制。
力矩限制来自 URDF，关节动力学由评测配置。
它与原 HoloMotion 的转速衰减模型仍有差异。

构建上下文按允许列表收集文件，排除测试和训练代码。
PyTorch 仅在构建时用于导出并检查标准观测计算图。
最终镜像通过 ONNX Runtime 执行这些图，并使用 Warp 缓冲区。
从源码执行时，若无编译好的图，仍使用原 PyTorch 后端。
编译包缺失部分内容、被修改或版本不匹配时，校验会失败。

`build-inputs.json` 记录源码、模型、资源和文件哈希。
`build-receipt.json` 记录镜像 ID 和展开后的体积。
通过 `docker save` / `docker load` 转移镜像时，请保留两份记录。
展开体积不能代表压缩传输大小或下载时长。

在项目环境中运行源码测试：
`python -m unittest discover -s holoboxing/evaluation/tests`。
在 `evaluation/viewer/` 中执行 `npm ci` 和 `npm run build` 构建播放器。
再分发前，须核查模型、机器人、擂台和依赖的适用许可。
完整性校验不授予再分发权。
播放器许可声明保留在 [evaluation/viewer/notices](viewer/notices)。

## 与已有工作的关系

底层控制器来自 [HoloMotion 项目](https://github.com/HorizonRobotics/HoloMotion)。
研究引用为 Chen 等的 [HoloMotion-1 Technical Report](https://arxiv.org/abs/2605.15336)，2026。
该报告介绍 V1；本评测器使用固定版本的 V2 模型包。
五点接口以本文定义为准。
比赛击打、倒地和耗时规则由本评测器定义，不是该论文的实验结论。

旋转表示引用 Zhou 等的[神经网络旋转表示连续性研究](https://arxiv.org/abs/1812.07035)，CVPR 2019。
具体 X/Z 排列和坐标系仍以比赛接口为准。
