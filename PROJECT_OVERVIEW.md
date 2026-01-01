# 项目交接说明（Point Hyperbolic Replay）

> 目的：帮助新同学快速理解项目背景、核心方法、训练策略、参数设置与当前效果。

## 1. 项目背景与目标

本项目配套硕士论文，研究**点云持续学习（Continual Learning）**，核心目标是：

* 在**任务不断新增**的场景下学习新的点云类别。
* 通过**双曲几何表示 + 经验回放**缓解灾难性遗忘。
* 在有限缓存与持续任务设置下提升稳定性与长期记忆能力。

该代码库主要围绕“**Point Hyperbolic Replay**”思想展开，训练对象为点云分类任务（如 ModelNet40 / ShapeNet）。

## 2. 核心方法概述

### 2.1 超曲空间特征表示

* **模型骨干**：`models/mlp.py` 中的 `HyperPointNet`
* 核心思想：将点云特征映射到 **Poincaré Ball 双曲空间**进行表示，捕获层级结构。
* 关键实现：
  * `ToPoincare` 映射
  * `_mobius_add` 组合特征
  * `get_hyper_graph_feature` 进行双曲特征构造

### 2.2 双曲三元组损失

* **函数位置**：`methods/MLP_correlation_CIL.py` 的 `hype_triplet_losses`
* 作用：约束 parent/child 结构在双曲空间中的距离关系，增强层级结构表达能力。

### 2.3 持续学习策略

* 入口脚本：`PointNet_CL_CIL.py`
* 训练过程按任务递增，每个任务增加 2 个类别。
* 模型输出层通过 `add_output_layer(2)` 扩展，适配新类别。
* 采用旧模型蒸馏（`ref_model`）与多种蒸馏损失约束当前模型。

### 2.4 回放缓存更新策略（Algorithm 1）

* 位置：`methods/MLP_correlation_CIL.py::update_memory`
* 名称：**Hyperbolic Manifold Expansion Elimination**
* 核心思路：
  * 从模型共享层提取局部/全局特征
  * 映射至切空间（`logmap0`）
  * 基于“局部/全局偏移”判断是否替换缓存样本

### 2.5 蒸馏与约束项

在增量任务阶段，训练 loss 由多个部分组成：

* 交叉熵（当前任务）
* 双曲三元组损失（层级结构）
* 旧模型蒸馏（`ref_model`）：
  * 余弦相似度约束 `mm_loss`
  * Wasserstein 距离 `w_loss`（已对点数抽样降低显存）
  * 特征匹配 `feature_loss`
* 输出层正交约束 `margin_loss`

## 3. 代码结构与关键入口

### 3.1 主要入口

* `PointNet_CL_CIL.py`：当前 CIL（Class-Incremental Learning）训练入口
* `PointNet_CL.py / PointNet_CL_1.py / PointNet_CL_2.py`：历史实验/消融入口

### 3.2 关键模块

* `models/mlp.py`：`HyperPointNet` 主干
* `methods/MLP_correlation_CIL.py`：核心训练策略与损失
* `pointCloud/utils/dataset.py`：ModelNet / ShapeNet 数据集加载
* `hyptorch/`：双曲数学与映射操作

## 4. 训练流程简述

以 `PointNet_CL_CIL.py` 为例：

1. 任务按 2 类递增拆分（`categories` 依次取子集）
2. 初始化模型（task0）后，后续任务扩展输出层
3. 每个任务训练 30 轮（`train()` 内）
4. 每轮训练：
   * 计算 CE + 三元组损失
   * 若存在旧模型（task>0），计算蒸馏损失
5. 任务结束后调用 `update_memory()` 更新缓存

## 5. 初始参数与默认配置

在 `PointNet_CL_CIL.py` 中默认参数（可调）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `num_task` | 20 | 任务数 |
| `categories` | 40类 | ModelNet40 类别顺序 |
| `mem_size` | 500 | 回放缓存大小 |
| `mem_batch` | 32 | 回放 batch |
| `train_batch_size` | 32 | 训练 batch |
| `val_batch_size` | 64 | 验证 batch |
| `eps_mem_batch` | 16 | 回放采样 batch |
| `dataroot` | `./modelnet40_normal_resampled` | 数据目录 |

## 6. 当前效果与记录

历史实验结果记录在：

* `readme.md`：包含多数据集结果记录（CIFAR/Imagenet/ModelNet 等）

目前 CIL 实验的完整对比记录仍需继续整理，建议统一写入 `result/` 或单独 CSV 文件，方便分析与绘图。

## 7. 快速运行建议

* 确保 ModelNet 数据解压到 `./modelnet40_normal_resampled`
* 直接运行：

```bash
python PointNet_CL_CIL.py
```

日志会写入 `./CIL_logs/`，模型权重保存在 `checkpoint/pointcloud/PointBest.pkl`。

## 8. 后续可能的改进方向（论文/毕业）

如果需要进一步改进效果，可考虑：

* 调整蒸馏项权重（`mm_loss / w_loss / feature_loss`）
* 尝试加入注意力蒸馏或关系蒸馏（已有模块）
* 调整回放替换策略超参（`alpha1/alpha2`）
* 将固定输出扩展策略与 Class-IL 头初始化做消融

---

如需继续维护，建议先阅读：

* `PointNet_CL_CIL.py`
* `methods/MLP_correlation_CIL.py`
* `models/mlp.py`
