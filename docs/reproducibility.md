# 复现指南

## 环境

原始验证环境为 Windows 与 Python 3.8：

```text
E:\anaconda3\envs\pytorch\python.exe
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如需尽量复现本次验证环境，可改用：

```powershell
python -m pip install -r requirements-lock.txt
```

## 从原始数据训练

将官方数据按 `data/README.md` 中的结构放到仓库根目录，然后运行：

```powershell
python src/train_full.py --project-root . --stage all
```

程序将依次：

1. 读取训练和测试批次，生成批次平均光谱；
2. 训练三组多尺度、五核 GPR，并生成五折 OOF；
3. 融合基础预测并拟合 C_mad3 煤种校准；
4. 提取 LVSE、Haar 和 PCA 几何表示；
5. 从训练 OOF 残差生成可靠性门控修正；
6. 写出固化权重、配置和两列提交文件。

也可以分阶段运行：

```powershell
python src/train_full.py --project-root . --stage data
python src/train_full.py --project-root . --stage base
python src/train_full.py --project-root . --stage final
```

## 使用固化模型生成提交

```powershell
python src/predict.py --weights model/model_weights.npz --output submit.csv
```

注意：固化权重保存了本次比赛测试批次的低维表示和基础模型输出，因此这个命令用于准确重建历史提交。对新的原始光谱推理时，应重新执行完整训练流程中的数据、基础模型和最终阶段。

## 校验

```powershell
python src/verify_package.py --package-root .
```

校验内容包括 UTF-8 编码、两列字段、批次唯一性、有限数值、行顺序以及预测结果与固化权重的一致性。

## 确定性边界

- 随机掩码通过固定种子控制。
- GPR 和 SVD 仍可能受到 NumPy、SciPy、scikit-learn、BLAS 实现和线程数差异影响。
- 原环境从零复跑与冻结提交的最大差异约为 `0.00005`，属于浮点舍入量级。
- 官方隐藏测试标签不可用，因此只能复现预测文件，不能在本地重算官方 RMSE。
