<div align="center">

<img src="assets/media/holomotion_logo_text.jpg" alt="HoloMotion Logo" width="600"/>

[![Safari](https://img.shields.io/badge/Website-006CFF?logo=safari&logoColor=fff)](https://horizonrobotics.github.io/robot_lab/holomotion/)
[![HuggingFace](https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=)](https://huggingface.co/collections/HorizonRobotics/holomotion)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HorizonRobotics/HoloMotion)
[![WeChat](https://img.shields.io/badge/Wechat-7BB32E?logo=wechat&logoColor=white)](https://horizonrobotics.feishu.cn/docx/Xs3cdEI8bo1EZuxUfzjckTgKn2c)
[![arXiv](https://img.shields.io/badge/arXiv-2605.15336-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.15336)
[![Bilibili](https://img.shields.io/badge/Bilibili-FB7299?logo=bilibili&logoColor=white)](https://space.bilibili.com/650643249)


</div>

# News
- [2026.07.16] HoloMotion v1.4 introduces HoloRetarget at 3,000+ FPS on RTX 4090 for training data generation and 300+ FPS entirely on-robot for teleoperation, plus HoloSMPL for 10+ datasets and devices.
- [2026.05.15] HoloMotion v1.3 scales from 60M to 0.4B parameters and 80 to 2000+ hours of motion data, while improving policy inference from ~100 to ~300 FPS.
- [2026.04.04] HoloMotion v1.2 provides pre-trained motion tracking and velocity tracking models for the community to deploy directly.


# Why HoloMotion?

## Larger, Faster and Stronger

HoloMotion scales humanoid whole-body control through a reference-conditioned MoE Transformer, large-scale motion data, and an optimized training-to-deployment pipeline, delivering stronger motion tracking with real-time inference efficiency.

<p align="center">
  <img src="assets/media/holomotion_overview.jpg" alt="HoloMotion overview" width="100%">
</p>


## Scales Toward 4-Any Humanoid Control

The roadmap of HoloMotion advances through four generalization targets, from motion imitation to command following, terrain adaptation, and embodiment transfer.

<p align="center">
  <img src="assets/media/holomotion_roadmap.jpg" alt="HoloMotion 4-Any roadmap" width="100%">
</p>

| Version  | Target Capability | Status      | Description                                                                                                                         |
| -------- | ----------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **v1.x** | Any Pose          | ✅ Done        | Achieve robust tracking and imitation of diverse, whole-body human motions, forming the core of the imitation learning capability.  |
| **v2.x** | Any Command       | 🚀 Next        | Enable language- and task-conditioned motion generation, allowing for goal-directed and interactive behaviors.                      |
| **v3.x** | Any Terrain       | 🧭 Planned    | Master adaptation to uneven, dynamic, and complex terrains, enhancing real-world operational robustness.                            |
| **v4.x** | Any Embodiment    | 🧭 Planned    | Generalize control policies across humanoids with varying morphologies and kinematics, achieving true embodiment-level abstraction. |

## Unifies Diverse Motion Sources

HoloSMPL standardizes VR, inertial, optical, and vision capture into a shared representation for HoloRetarget.

<p align="center">
  <img src="assets/media/holosmpl_capture_sources.jpg" alt="HoloSMPL inputs from VR, inertial, optical, and vision capture" width="100%">
</p>

## Closes the Loop From Data to Robots

HoloMotion provides a clear, modular framework for bridging motion data, policy learning, simulation evaluation, and real-robot deployment.

<p align="center">
  <img src="assets/media/holomotion_pipeline.jpg" alt="HoloMotion pipeline" width="100%">
</p>

## No Per-User Training Required

Whether you want to replay motions, stream live teleoperation, or train a custom policy, HoloMotion provides a direct path into the workflow:

| User Goal | Start Here | What You Need |
| --------- | ---------- | ------------- |
| **Offline motion tracking**<br/>Replay local motion clips for demos such as dance or scripted performances. | [Real-world deployment: Offline Motion](docs/realworld_deployment.md#offline-motion-tracking) | The v1.4 Docker image and retargeted `.npz` motion clips. No model training is required. |
| **Online motion tracking**<br/>Follow live VR or teleoperation motion streams. | [Real-world deployment: Teleoperation](docs/realworld_deployment.md#live-teleoperation) | The v1.4 Docker image, robot deployment setup, and a live motion source. No model training is required. |
| **Train your own model**<br/>Build a custom policy from your own motion data. | [Environment setup](docs/environment_setup.md) → [HoloSMPL conversion](holosmpl/README.md) → [Retargeting](docs/motion_retargeting.md) → [Training](docs/train_motion_tracking.md) → [Evaluation](docs/evaluate_motion_tracking.md) | Training environment, curated motion data, retargeted HDF5 datasets, and GPU resources. |

# Join Us

We are hiring full-time engineers, new graduates, and interns who are excited about humanoid robots, motion control, and embodied intelligence.
Send your resume by scanning the **WeChat** QR code below to get in touch with us.

<p align="center">
  <img width="420" height="150" src="assets/media/qr_codes.jpg" hspace="10">
</p>

# Projects Using HoloMotion

We are glad to see HoloMotion being used as a motion control foundation for humanoid research and applications.

| Project | Description | Links |
| --- | --- | --- |
| OMG: Omni-Modal Motion Generation for Generalist Humanoid Control | Uses HoloMotion as the motion control foundation for omni-modal motion generation and generalist humanoid control. | [Paper](https://arxiv.org/abs/2606.10340) / [Project](https://tsinghua-mars-lab.github.io/OMG/) / [Code](https://github.com/Tsinghua-MARS-Lab/OMG) |
| HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory | Uses HoloMotion as the humanoid whole-body motion skill within its closed-loop Embodied AgentOS. | [Paper](https://arxiv.org/abs/2606.23565) / [Project](https://horizonrobotics.github.io/robot_lab/holoagent/) / [Code](https://github.com/HorizonRobotics/HoloAgent) |

If your project uses HoloMotion, feel free to open a PR to add it here.

# Citation

```
@misc{chen2026holomotion1,
  title = {HoloMotion-1 Technical Report},
  author = {Maiyue Chen and Kaihui Wang and Bo Zhang and Xihan Ma and Zhiyuan Yang and Yi Ren and Qijun Huang and Zihao Zhu and Yucheng Wang and Zhizhong Su},
  year = {2026},
  eprint = {2605.15336},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2605.15336}
}
```

# Acknowledgements

This project is built upon and inspired by several outstanding open source projects:

- [ASAP](https://github.com/LeCAR-Lab/ASAP)
- [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)
- [GMR](https://github.com/YanjieZe/GMR)
- [GVHMR](https://github.com/zju3dv/GVHMR)
- [Humanoidverse](https://github.com/LeCAR-Lab/HumanoidVerse)
- [Mink](https://github.com/kevinzakka/mink)
- [MotionMillion](https://github.com/VankouF/MotionMillion-Codes)
- [PBHC](https://github.com/TeleHuman/PBHC)
- [PHC](https://github.com/ZhengyiLuo/PHC?tab=readme-ov-file)
- [ProtoMotion](https://github.com/NVlabs/ProtoMotions/tree/main/protomotions)
- [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
