# 模型文件

- `model_weights.npz`：最终固化权重和历史测试批次推理所需数组。
- `base_component_predictions.npz`：完整训练产生的多尺度基础分支 OOF 与测试输出。
- `model_config.json`：固定配置、训练 OOF 指标、煤种校准参数和文件哈希。

这些文件用于复现官方 RMSE `140.71316` 对应的历史提交。原始或批次平均光谱不包含在本目录中。
