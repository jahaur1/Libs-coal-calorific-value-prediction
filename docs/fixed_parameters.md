# 固定参数说明

最优官方提交是比赛结束时的历史快照，因此代码中保留了一组固定配置。固定不等于测试样本编号修补：这些参数描述光谱处理、模型容量、融合和残差收缩方式，对所有样本统一生效。

| 参数 | 作用 |
|---|---|
| `SEED` | 固定随机掩码扰动，保证重复训练可比较 |
| `FFT_CUTOFF` | 去除光谱最低频背景的比例 |
| `VIP_THRESHOLD` | PLS-VIP 波长选择阈值 |
| `VIP_COMPONENTS` | 计算 VIP 时使用的最大 PLS 分量数 |
| `GPR_RESTARTS` | GPR 核超参数优化的重启次数 |
| `MASK_AGGREGATES` | 全训练集波长掩码扰动的集成次数 |
| `FIXED_PAIRS` | SG 导数窗口与局部邻域半径的多尺度组合 |
| `PAIR_WEIGHTS` | 三个尺度分支的固定融合权重 |
| `KERNEL_WEIGHTS` | 五种 GPR 核的固定融合权重 |
| `RAW_POINTS` | 几何残差表示的统一波长采样点数 |
| `EMISSION_BANDS` | 构建局部连续谱残差和 Haar 特征的物理波段 |
| `CORRECTION_SCALE` | 对邻域残差修正进行保守收缩 |
| `GEOMETRY_COMPONENTS` | 可靠性门控使用的低维几何分量数 |

训练生成的校准斜率、截距、异常训练行、批次标识和期望预测并不是手工常量，而是由输入训练数据和固定流程计算后写入 `model/model_weights.npz` 或 `model/model_config.json` 的固化结果。
