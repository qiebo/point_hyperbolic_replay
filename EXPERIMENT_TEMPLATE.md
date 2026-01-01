# CIL 实验对比模板（Herding 缓存）

> 目标：对比 “原缓存策略” vs “Herding 类原型缓存”，验证 ACC 提升。

## 1. 实验设置

* 数据集：ModelNet40（默认）
* 任务划分：每任务 2 类，`num_task = 20`
* 训练入口：`PointNet_CL_CIL.py`
* 固定随机种子：`setup_seed(114514)`（如需对比，可改多个 seed）
* 主要指标：
  * 平均准确率（Average Acc）
  * 最终任务准确率（Final Acc）
  * 遗忘率（如可计算 BWT）

## 2. 对比项

| 实验编号 | 缓存策略 | 备注 |
| --- | --- | --- |
| A | 原缓存策略 | 当前基线 |
| B | Herding（类原型选择） | 本次改进 |

## 3. 运行命令

```bash
python PointNet_CL_CIL.py
```

> 建议每个实验至少跑 3 个 seed，报告均值与方差。

## 4. 日志与结果记录

* 日志位置：`./CIL_logs/`
* 模型权重：`checkpoint/pointcloud/PointBest.pkl`

建议额外保存：

* 每 task 的 Acc
* Average Acc / Final Acc
* 训练时间

可以将结果写到 CSV（自行补充脚本）。

## 5. 论文撰写建议

可在“消融实验”中说明：

* Herding 在持续学习中更稳定的类原型代表性
* 与原策略相比，提升了旧类保持能力（表现为 Acc 提升/遗忘降低）
