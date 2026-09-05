# Field LIBS Coal Calorific Value Prediction

基于现场 LIBS 光谱的煤炭弹筒发热量预测方案。本仓库用于归档参赛期间的建模思路、完整训练与推理代码、固化模型以及官方成绩。

> 本仓库是个人参赛记录，不代表上海海事大学、科大讯飞或赛事官方。原始竞赛数据不在仓库中提供。

## 比赛信息

| 项目 | 内容 |
|---|---|
| 赛事 | 基于现场 LIBS 光谱的煤炭弹筒发热量预测挑战赛 |
| 举办方 | 上海海事大学 |
| 赛事平台 | 科大讯飞 AI 开发者大赛 |
| 正赛时间 | 2026-06-09 至 2026-08-27 |
| 参赛团队 | 2020 |
| 奖金池 | ￥10,000 |
| 任务 | 多实例、弱对齐、时间外推回归 |
| 官方指标 | RMSE，越低越好 |
| 最优官方得分 | **140.71316** |

同一批次包含十至二十余条现场瞬态 LIBS 光谱，但只对应一个实验室混合煤样的宏观发热量标签。训练集与测试集又按照月份沿时间轴切分，因此主要难点是批次内光谱聚合、弱对齐噪声和跨时间分布偏移。

## 最终方法

最终方案由以下部分组成：

1. 将同一批次的多条瞬态光谱求稳健批次表示，避免把共享标签的光谱误当成相互独立的样本。
2. 使用 FFT 高通、Savitzky–Golay 一阶导数与 MSC 抑制基线漂移和散射差异。
3. 通过训练标签上的 PLS-VIP 选择有效波长，并在多个邻域尺度构建局部平滑特征。
4. 融合 RBF、Matérn、Rational Quadratic 和 Dot Product 等多个 GPR 核。
5. 使用训练集折外残差执行 C_mad3 稳健筛选与煤种内留一法线性校准。
6. 提取 LVSE、局部发射带 Haar 多尺度能量和 PCA 几何表示，用于样本级残差可靠性门控。
7. 对每个煤种的残差修正保持均值为零，减少校准后整体均值漂移。

实现细节见 [建模历程](docs/modeling_journey.md) 和 [固定参数说明](docs/fixed_parameters.md)。

## 官方成绩记录

| 阶段 | 官方 RMSE | 结果 |
|---|---:|---|
| GPR + FFT + SG1 + MSC + 煤种校准 | 174.41325 | 早期可靠基线 |
| C_mad3 稳健多核校准 | 143.81585 | 显著提升 |
| 多视图残差共识候选 | 141.41519 | 未超过最优方案 |
| 多尺度多核 GPR + 可靠性门控残差修正 | **140.71316** | **最终最优** |

这里只将比赛平台实际返回的数值称为“官方成绩”。训练集 OOF 指标用于模型开发，不能与隐藏测试集 RMSE 直接等同。结构化记录见 [official_scores.csv](docs/official_scores.csv)。

## 仓库结构

```text
.
├─ src/                         # 完整训练、推理和校验代码
├─ model/                       # 最优方案的固化模型与配置
├─ docs/                        # 赛事、建模历程和复现说明
├─ data/README.md               # 数据目录约定，不含原始数据
├─ submit.csv                   # 140.71316 对应的两列预测结果
├─ CHECKSUMS.sha256             # 核心结果文件完整性校验
└─ requirements.txt
```

## 快速复现

建议使用 Python 3.8 或兼容环境：

```powershell
python -m pip install -r requirements.txt
python src/train_full.py --project-root . --stage all
python src/predict.py
python src/verify_package.py
```

完整数据目录、分阶段训练方法和确定性说明见 [复现指南](docs/reproducibility.md)。从零训练会重新生成 `training_artifacts/prepared_data.npz`，该文件是原始数据的派生中间件，已被 Git 忽略。

## 数据与合规说明

- 仓库不包含训练集、测试集、标签表或由完整光谱直接导出的训练中间件。
- 代码中的标签读取路径仅指向训练标签；测试光谱用于冻结模型后的特征变换与推理。
- `model/` 和 `submit.csv` 是比赛结束后的历史结果快照，不应被解释为公开数据集。
- 使用者需要自行从赛事官方渠道获取数据，并遵守赛事数据许可与平台规则。
- 仓库没有设置开源许可证；未经明确授权，不应假定代码或赛事材料可被任意再分发。

## 运行环境

原始复跑使用：

```text
E:\anaconda3\envs\pytorch\python.exe
```

主要依赖为 NumPy、pandas、SciPy、scikit-learn、openpyxl 和 threadpoolctl。`requirements.txt` 提供兼容版本范围，`requirements-lock.txt` 记录本次成功复跑的精确版本。
