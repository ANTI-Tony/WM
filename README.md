# CausalComp: 基于模块化因果交互发现的组合泛化世界模型

> **Compositional Generalization in World Models via Modular Causal Interaction Discovery**

## 核心思想

现有世界模型的两个阵营各有缺陷：
- **因果世界模型**（STICA, OOCDM）学到了因果结构，但是整体式的，无法泛化到新的物体组合
- **组合世界模型**（DreamWeaver, FIOC-WM）学到了模块化表征，但没有因果推理，无法区分伪相关

**CausalComp** 将两者统一：从视频中自动发现物体间的因果交互图，为每种交互类型学习独立的动态模块，从而实现对**从未见过的物体-交互组合**的零样本动态预测。

## 架构

```
输入视频帧 → [Slot Attention 物体发现]
                    ↓
            K 个物体 slots
                    ↓
           [因果交互图发现]  ← 核心创新 1
            · 边存在性预测
            · 干预性验证 (mask slot j, 观察 slot i 是否变化)
            · 交互类型分类 (Gumbel-Softmax 路由)
                    ↓
            因果图 G = (V, E, τ)
                    ↓
           [模块化因果动态]  ← 核心创新 2
            · f_self: 物体自演化 (惯性, 重力)
            · f_inter[τ]: 类型 τ 专属交互模块 (碰撞/接触/推拉/...)
            · 模块可跨新组合复用 → 组合泛化
                    ↓
            预测下一时刻 slots
                    ↓
           [空间广播解码器] → 重建图像
```

## 项目结构

```
CausalComp/
├── configs/default.py          # 超参数配置
├── models/
│   ├── slot_attention.py       # Slot Attention 物体发现
│   ├── causal_graph.py         # 因果图发现 + 交互类型分类
│   ├── modular_dynamics.py     # 类型化 GNN 动态模块
│   ├── decoder.py              # 空间广播解码器
│   └── causalcomp.py           # 主模型 (整合所有模块)
├── data/clevrer_dataset.py     # CLEVRER 数据集加载器
├── train.py                    # 训练脚本
├── test_smoke.py               # 冒烟测试
├── utils/
│   ├── logger.py               # 日志工具
│   └── visualize.py            # 可视化 (slot 分解, 因果图, 轨迹)
└── scripts/
    ├── setup_runpod.sh         # RunPod 一键部署
    └── download_clevrer.sh     # CLEVRER 数据下载
```

## 快速开始

### 本地冒烟测试（无需 GPU）

```bash
pip install torch torchvision einops
python test_smoke.py
```

### RunPod 训练

```bash
git clone https://github.com/ANTI-Tony/WM.git CausalComp
cd CausalComp
bash scripts/setup_runpod.sh          # 安装依赖 + 下载 CLEVRER + 冒烟测试

# Debug 训练
python train.py --exp_name debug --num_epochs 5 --batch_size 8 --resolution 64

# 完整训练
python train.py --exp_name v1_full --wandb
```

## 实验设计

### 数据集：CLEVRER-Comp

基于 [CLEVRER](http://clevrer.csail.mit.edu/) 设计组合泛化 split（MCD 方法）：
- **训练集**：60% 的物体-交互类型组合
- **测试集-已见**：训练组合的 held-out 样本
- **测试集-新组合**：40% 训练中从未出现的组合

### 评估指标

| 指标 | 说明 |
|------|------|
| Seen Acc | 已见组合上的动态预测准确率 |
| Unseen Acc | 新组合上的动态预测准确率 |
| **Harmonic Mean** | 2×S×U/(S+U)，**主指标** |
| Comp Gap | Seen - Unseen，越小越好 |
| Graph F1 | 因果图发现的准确率 |
| Counterfactual Acc | 反事实预测准确率 |

### 基线

| 方法 | 会议 | 因果 | 组合 |
|------|------|:----:|:----:|
| DreamerV3 | Nature 2025 | ✗ | ✗ |
| SlotFormer | ICLR 2023 | ✗ | ✗ |
| OOCDM | ICML 2024 | ✓ | ✗ |
| DreamWeaver | ICLR 2025 | ✗ | ✓ |
| FIOC-WM | NeurIPS 2025 | ✗ | ✓ |
| Causal-JEPA | arXiv 2026 | ✓ | ✗ |
| **CausalComp (Ours)** | — | **✓** | **✓** |

## 相关工作

- [Slot Attention](https://arxiv.org/abs/2006.15055) (NeurIPS 2020)
- [Interaction Networks](https://arxiv.org/abs/1612.00222) (NeurIPS 2016)
- [Robust Agents Learn Causal World Models](https://arxiv.org/abs/2402.10877) (ICLR 2024 Oral)
- [DreamWeaver](https://arxiv.org/abs/2501.14174) (ICLR 2025)
- [FIOC-WM](https://arxiv.org/abs/2511.02225) (NeurIPS 2025)
- [Causal-JEPA](https://arxiv.org/abs/2602.11389) (arXiv 2026)
- [CLEVRER](http://clevrer.csail.mit.edu/) (ICLR 2020)
