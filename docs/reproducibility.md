# 复现指南

## 运行环境

已验证环境：

```text
Windows 10 10.0.26200
Python 3.8.19
CPU 推理，无需 CUDA、CuDNN、PyTorch 或编译扩展
```

安装固定版本依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 数据放置

将赛事数据放入根目录的 `xfdata/`。程序兼容两种形式：数据目录直接位于 `xfdata/`，或者位于 `xfdata/` 下唯一一层数据集目录中。

```text
xfdata/
├─ 训练集/
├─ 训练集标签/
├─ 测试集/
└─ submit_sample/
   └─ .../submit.csv
```

## 固化模型推理

Linux 或 Git Bash：

```bash
bash test.sh
```

如果 Python 命令不是 `python3`：

```bash
PYTHON_BIN=python bash test.sh
```

Windows PowerShell 可直接运行：

```powershell
E:\anaconda3\envs\pytorch\python.exe code\predict.py `
  --xfdata-root xfdata `
  --weights user_data\model_data\model_weights.npz `
  --output prediction_result\result
```

推理会从原始训练和测试光谱重建批次特征，在冻结的 VIP 掩码、核参数、融合权重和校准参数下重建 GPR 状态，最终生成 `prediction_result/result`。

## 完整训练

```bash
bash train.sh
```

训练依次完成：

1. 读取 70 个训练批次和 26 个测试批次并生成批次均值光谱；
2. 对 7 个尺度和 5 类核生成 35 个基础分支及训练折外预测；
3. 使用训练集内层验证选择正则强度并学习非负融合权重；
4. 执行 C_mad3 煤种内稳健校准；
5. 从训练环境验证选择几何维数和残差修正比例；
6. 写入 `user_data/model_data/` 并生成 `prediction_result/result`。

训练中间文件写入 `user_data/tmp_data/`，不进入 Git 版本控制。

## 完整性检查

```bash
python3 code/verify_package.py --xfdata-root xfdata
```

检查内容包括官方目录结构、UTF-8 编码、两列字段、有限预测值、测试标识顺序，以及固化权重中是否出现测试专用预测数组。

当前参考结果包含 26 行，SHA-256 为：

```text
903637a78e67f5aee3b82fb443a7d6d24e5524f4453fbafa61c018e1ba8a04f8
```

## 确定性边界

- 随机种子、候选集合、数据划分与线程限制均固定在源码中。
- NumPy、SciPy、scikit-learn 或底层 BLAS 实现不同，仍可能产生浮点舍入量级差异。
- 官方隐藏标签不可用，因此本地只能验证训练、推理和预测文件一致性，不能重新计算平台 RMSE。
