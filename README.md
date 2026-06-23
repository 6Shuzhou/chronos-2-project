# Chronos-2 Time Series Forecasting Project

本项目基于 **Amazon Chronos-2** 预训练模型，使用 **AutoGluon TimeSeries** 原生 API 进行**零样本预测**和**微调迁移学习**，在 **ETT** 数据集上进行测试。

代码遵循 AutoGluon 官方教程：[Forecasting with Chronos-2](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)

## 项目结构

```
chronos-2-project/
├── configs/
│   └── config.yaml              # 配置文件
├── data/                        # 数据目录
├── models/                      # 模型保存目录（实验后自动清理）
├── notebooks/                   # Jupyter notebooks
├── results/                     # 结果输出目录
│   └── plots/                   # 可视化图表
├── src/
│   ├── __init__.py
│   ├── ett_loader.py            # ETT数据集加载器
│   ├── chronos2_experiment.py   # 核心实验代码（零样本/微调/对比）
│   ├── evaluate.py              # 评估与可视化工具
│   └── main.py                  # 主入口
├── requirements.txt
└── README.md
```

## 环境要求

- Python >= 3.9
- CUDA (optional, for GPU acceleration)

## 安装

```bash
pip install -r requirements.txt
```

核心依赖：
- `autogluon.timeseries[all]` - AutoGluon时间序列框架（内置Chronos-2支持）
- `chronos-forecasting` - Amazon Chronos模型库
- `huggingface-hub` - 模型下载
- `pandas`, `numpy`, `matplotlib` - 数据处理与可视化

## 数据

ETT (Electricity Transformer Temperature) 数据集已下载至 `/root/datasets/ett-dataset/`：
- `ETTh1.csv` / `ETTh2.csv` - 小时级数据
- `ETTm1.csv` / `ETTm2.csv` - 15分钟级数据

## 模型

Chronos-2 预训练模型已下载至 `/root/chronos-2-model/`。

## 使用方式

### 1. 零样本预测 (Zero-Shot)

直接使用预训练的 Chronos-2 模型进行预测，无需微调：

```bash
cd /root/chronos-2-project
python src/main.py --mode zero_shot --dataset ETTh1 --prediction_length 96 --plot
```

AutoGluon 代码示例：
```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(prediction_length=96).fit(
    train_data,
    hyperparameters={"Chronos2": {"model_path": "/root/chronos-2-model"}},
    enable_ensemble=False,
)
predictions = predictor.predict(train_data)
```

### 2. 微调/迁移学习 (Fine-tuning)

在特定数据集上微调 Chronos-2 模型。默认使用 **LoRA** 适配器，占用空间极小：

```bash
python src/main.py --mode finetune --dataset ETTh1 --prediction_length 96 \
    --time_limit 600 --fine_tune_mode lora --plot
```

AutoGluon 代码示例：
```python
predictor = TimeSeriesPredictor(prediction_length=96).fit(
    train_data,
    hyperparameters={
        "Chronos2": {
            "model_path": "/root/chronos-2-model",
            "fine_tune": True,
            "fine_tune_mode": "lora",  # 或 "full"
        }
    },
    enable_ensemble=False,
)
```

### 3. 对比实验

同时运行零样本和微调模型，自动生成对比表格：

```bash
python src/main.py --mode compare --dataset ETTh1 --prediction_length 96 \
    --time_limit 900 --plot
```

AutoGluon 代码示例：
```python
predictor = TimeSeriesPredictor(prediction_length=96).fit(
    train_data,
    hyperparameters={
        "Chronos2": [
            {"model_path": "/root/chronos-2-model", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "/root/chronos-2-model", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    enable_ensemble=False,
)
leaderboard = predictor.leaderboard(test_data)
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `zero_shot` | 运行模式: zero_shot / finetune / compare |
| `--dataset` | `ETTh1` | 数据集: ETTh1 / ETTh2 / ETTm1 / ETTm2 |
| `--prediction_length` | `96` | 预测步长 |
| `--model_path` | `/root/chronos-2-model` | 模型路径 |
| `--time_limit` | `300` | 训练时间限制（秒） |
| `--fine_tune_mode` | `lora` | 微调模式: lora / full |
| `--fine_tune_lr` | `1e-3` | 微调学习率 |
| `--fine_tune_steps` | `1000` | 微调步数 |
| `--plot` | - | 生成预测可视化 |

## 空间节省说明

**模型文件不保留**：实验结束后自动清理 AutoGluon 生成的模型目录，只保留：
- `results/*.json` - 实验结果（指标、超参数等）
- `results/*.csv` - 预测值
- `results/plots/*.png` - 可视化图表

如需保留模型，可修改 `src/chronos2_experiment.py` 中的 `cleanup_model_dir()` 调用。

## ETT数据加载

```python
from src.ett_loader import prepare_ett_for_chronos

train, val, test = prepare_ett_for_chronos(
    dataset_name="ETTh1",
    prediction_length=96,
    data_dir="/root/datasets/ett-dataset",
)
```

## 参考

- [AutoGluon Time Series Documentation](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)
- [Forecasting with Chronos-2 Tutorial](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)
- [Chronos-2 on HuggingFace](https://huggingface.co/amazon/chronos-2)
- [Chronos Forecasting GitHub](https://github.com/amazon-science/chronos-forecasting)
- [ETT Dataset](https://github.com/zhouhaoyi/ETDataset)
