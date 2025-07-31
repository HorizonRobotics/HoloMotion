<div align="center">

<img src="assets/media/holomotion_logo_text.png" alt="HoloMotion Logo" width="500"/>

---

[![Safari](https://img.shields.io/badge/Website-006CFF?logo=safari&logoColor=fff)](https://horizonrobotics.github.io/robot_lab/holomotion/)
[![Python](https://img.shields.io/badge/Python3.8-3776AB?logo=python&logoColor=fff)](#)
[![Ubuntu](https://img.shields.io/badge/Ubuntu22.04-E95420?logo=ubuntu&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?logo=apache&logoColor=white)](./LICENSE)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2025.00000-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2025.00000) -->

</div>

## Introduction

**HoloMotion: A Foundation Model for Whole-Body Humanoid Motion Tracking**

HoloMotion is a unified foundation model designed for robust, real-time, and generalizable whole-body tracking of humanoid robots. Built upon high-fidelity motion capture data and advanced retargeting techniques, HoloMotion bridges the gap between human motion and embodied robot control across diverse terrains and morphologies.

The framework supports the complete pipeline from data preparation to real-world deployment, including motion retargeting, distributed training with reinforcement learning and motion priors, performance evaluation, and ROS2-based deployment. With modular design, HoloMotion enables humanoid agents to imitate, and generalize whole-body motions in both simulation and the physical world.

If you're building a next-generation humanoid controller in embodied systems, HoloMotion offers a practical and extensible foundation for whole-body motion tracking.

---

### 🛠️ Roadmap: Progress Toward Any Humanoid Control

We envision HoloMotion as a general-purpose foundation for humanoid motion tracking and control. Its development is structured around four core generalization goals: Any Pose, Any Command, Any Terrain, and Any Embodiment. Each goal corresponds to a major version milestone.

| Version   | Target Capability     | Description |
|-----------|------------------------|-------------|
| **v0.2.x** | 🧪 Alpha Preview        | Infrastructure for motion retargeting, training, and deployment to support future Any Pose capability |
| **v1.0**   | 🔄 Any Pose             | Robust tracking and imitation of diverse whole-body human motions |
| **v2.0**   | ⏳ Any Command          | Language- and task-conditioned motion generation and control |
| **v3.0**   | ⏳ Any Terrain          | Adaptation to uneven, dynamic, and complex real-world environments |
| **v4.0**   | ⏳ Any Embodiment       | Generalization across humanoids with different morphologies and kinematics |

> Each stage builds on the previous one, moving from motion imitation to instruction following, terrain adaptation, and embodiment-level generalization.

## Pipeline Overview

```mermaid
flowchart LR
    A["🔧 1. Environment Setup<br/>Dependencies & conda"]

    subgraph dataFrame ["DATA"]
        B["📊 2. Dataset Preparation<br/>Download & curate"]
        C["🔄 3. Motion Retargeting<br/>Human to robot motion"]
        B --> C
    end

    subgraph modelFrame ["TRAIN & EVAL"]
        D["🧠 4. Model Training<br/>Train with HoloMotion"]
        E["📈 5. Evaluation<br/>Test & export"]
        D --> E
    end

    F["🚀 6. Deployment<br/>Deploy to robots"]

    A --> dataFrame
    dataFrame --> modelFrame
    modelFrame --> F

    classDef subgraphStyle fill:#f9f9f9,stroke:#333,stroke-width:2px,stroke-dasharray:5 5,rx:10,ry:10,font-size:16px,font-weight:bold
    classDef nodeStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,rx:10,ry:10
    
    class dataFrame,modelFrame subgraphStyle
    class A,B,C,D,E,F nodeStyle
```

## Quick Start

### 🔧 1. Environment Setup [[Doc](docs/environment_setup.md)]

Follow the instructions in the documentation to create two conda environments:

- `holomotion_train`: For training and evaluation.
- `holomotion_deploy`: For ROS2 deployment in real-world scenarios.

If you only intend to use our pretrained models, you can skip the training environment setup and proceed directly to configure the deployment environment. See the [real-world deployment documentation](docs/realworld_deployment.md) for details.

### 📊 2. Dataset Preparation [[Doc](docs/smpl_data_curation.md)]

Download motion capture datasets and convert them into AMASS-compatible format. Our repository includes preliminary data filtering capabilities to remove abnormal data based on kinematic metrics.

### 🔄 3. Motion Retargeting [[Doc](docs/motion_retargeting.md)]

Convert AMASS-compatible SMPL data into robot-specific motion sequences. Our pipeline currently supports **[PHC](https://github.com/ZhengyiLuo/PHC?tab=readme-ov-file)** and **[Mink](https://github.com/kevinzakka/mink)** retargeting methods, with additional methods planned for future releases.

### 🧠 4. Model Training [[Doc](docs/train_motion_tracking.md)]

Package the retargeted motion data into a training-friendly LMDB database and initiate distributed training across multiple GPUs. We support multiple training paradigms including:

- **PPO**: Pure reinforcement learning
- **AMP**: Adversarial motion prior training
- **DAgger** (optionally with PPO): Teacher-student distillation training

### 📈 5. Evaluation [[Doc](docs/evaluate_motion_tracking.md)]

Visualize and evaluate model performance using widely adopted metrics, then export validated models for deployment. For detailed metric definitions, please refer to the [evaluation documentation](docs/evaluate_motion_tracking.md#evaluation-results).

### 🚀 6. Real-world Deployment [[Doc](docs/realworld_deployment.md)]

Deploy the exported ONNX model using our ROS2 package to run on real-world robots.

## Citation

```
@software{holomotion_2025,
  author = {Maiyue Chen, Kaihui Wang, Bo Zhang, Yi Ren, Zihao Zhu, Yucheng Wang, Zhizhong Su},
  title = {HoloMotion: A Foundation Model for Whole-Body Humanoid Motion Tracking},
  year = {2025},
  month = july,
  version = {0.2.2},
  url = {https://github.com/HorizonRobotics/HoloMotion},
  license = {Apache-2.0}
}
```

## License

This project is released under the **[Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)** license.

## Acknowledgements

This project is built upon and inspired by several outstanding open source projects:

- [ASAP](https://github.com/LeCAR-Lab/ASAP)
- [Humanoidverse](https://github.com/LeCAR-Lab/HumanoidVerse)
- [PHC](https://github.com/ZhengyiLuo/PHC?tab=readme-ov-file)
- [ProtoMotion](https://github.com/NVlabs/ProtoMotions/tree/main/protomotions)
- [Mink](https://github.com/kevinzakka/mink)
- [PBHC](https://github.com/TeleHuman/PBHC)
