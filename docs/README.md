# 多模态活动预测系统

## 项目简介

这是一个基于多模态信息融合的活动预测系统，支持静态预测（总体表现）和动态预测（时间序列表现）。系统采用WXM模型架构，融合文本、数值特征和时间特征，通过四通路跨模态注意力机制实现高效预测。项目提供统一的数据处理框架、可配置的模型架构以及完整的API服务接口。

## 项目结构

```
├── api_service/         # API服务模块
│   ├── predict_api.py   # Flask API服务器
│   ├── predict_client.py # 命令行客户端
│   ├── examples/        # 示例数据
│   │   ├── static_example.csv   # 静态预测示例
│   │   └── dynamic_example.csv  # 动态预测示例
│   └── README.md        # API服务使用文档
├── configs/             # 🔧 配置文件
│   └── config.yaml      # 主配置文件
├── data/                # 📊 数据目录
│   ├── activity_order_independent_id/  # 原始订单序列数据
│   ├── activities_duration_*.csv       # 活动持续时间数据
│   ├── all_activities_attributes*.csv  # 活动属性数据
│   └── dataset_df.csv   # 处理后的数据集
├── models/              # 🤖 模型定义和权重
│   ├── WXM.py          # 主模型（多模态融合）
│   ├── DDN.py          # DDN归一化模块
│   ├── WXM_static.pth  # 静态预测模型权重
│   ├── WXM_dynamic.pth # 动态预测模型权重
│   └── last_args.json  # 模型参数记录
├── utils/               # 🛠️ 工具模块
│   ├── data_processor.py    # 统一数据处理
│   ├── static_exp.py        # 静态预测实验
│   ├── dynamic_exp.py       # 动态预测实验
│   ├── tools.py             # 通用工具函数
│   ├── ADF.py               # ADF测试工具
│   └── learnable_wavelet.py # 可学习小波变换
├── layers/              # 网络层
│   ├── RevIN.py        # 可逆归一化
│   └── fds.py          # 特征分布标准化
├── results/             # 📈 结果输出
│   ├── static/         # 静态预测结果
│   └── dynamic/        # 动态预测结果
├── scripts/             # 脚本目录
│   ├── cleanup.py      # 项目清理脚本
│   └── test_parameter_compatibility.py  # 参数兼容性测试
├── test_ipynb/         # 测试和实验notebook
├── docs/               # 文档目录
├── bert-base-uncased/  # BERT模型文件
├── assets/             # 资源文件（图片等）
├── train_new.py        # 🚀 主训练入口
├── train.py            # 原有训练脚本（兼容性保留）
└── requirements.txt    # 依赖包
```

**关键目录说明：**
- 🔧 **configs/** - 配置文件目录
- 📊 **data/** - 数据目录  
- 🤖 **models/** - 模型定义和权重目录
- 🛠️ **utils/** - 工具模块目录
- 📈 **results/** - 结果输出目录
- 🚀 **train_new.py** - 主训练入口文件

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 确保BERT模型文件存在
# 项目已包含bert-base-uncased模型文件
```

### 2. 配置参数
编辑 `configs/config.yaml` 文件，调整数据路径、模型参数等：
```yaml
data:
  static_target: 'duration'  # 静态预测目标
  dynamic_dataset_path: 'data/dataset_df.csv'
  
model:
  model_type: 'WXM'
  embedding_dim: 256
  hidden_dim: 512
  
training:
  static_epochs: 50
  dynamic_epochs: 100
```

### 3. 训练模型
```bash
# 静态预测（预测活动总体表现）
python train_new.py --config configs/config.yaml --task_type static

# 动态预测（预测时间序列表现）
python train_new.py --config configs/config.yaml --task_type dynamic

# 如果需要重新处理数据
python train_new.py --config configs/config.yaml --task_type static --process_data
```

### 4. 使用API服务

#### 启动API服务器
```bash
cd api_service
python predict_api.py
```

#### 使用命令行客户端
```bash
# 静态预测
python predict_client.py --csv_file examples/static_example.csv --task_type static --output_file result.csv

# 动态预测
python predict_client.py --csv_file examples/dynamic_example.csv --task_type dynamic --output_file result.csv
```

#### REST API调用
```bash
# 获取API信息
curl http://localhost:5000/api/info

# 获取所需列名
curl http://localhost:5000/api/columns

# 静态预测
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"task_type": "static", "data": [...]}'

# 动态预测
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"task_type": "dynamic", "data": [...]}'
```

## 数据格式

### 必需字段
- **数值特征**: `activity_budget`, `max_reward_count`, `min_reward_count`, `duration`
- **文本特征**: `activity_name`, `activity_title`, `product_names`
- **标签特征**: `customer_id`, `template_id`, `activity_type`, `activity_form`, `bank_name`, `location`, `main_reward_type`, `secondary_reward_type`, `threshold`
- **时间特征**: `day`, `week`, `month`, `year`
- **序列特征** (动态预测): `0`, `1`, `2`, ..., `20` (21个时间点的数值)

### 示例数据
参考 `api_service/examples/` 目录中的示例文件。

## 配置说明

主要配置项：
- `data`: 数据路径和目标变量配置
- `model`: WXM模型架构参数
- `features`: 多模态特征使用配置
- `training`: 训练超参数（学习率、批次大小等）
- `sequence`: 序列长度和处理参数
- `ddn`: DDN归一化参数（可选）

详细配置说明请参考 `docs/快速开始指南.md`

## 实验结果

### 性能指标
- **静态预测**: 分位数损失、MAE、MSE
- **动态预测**: MAE、MSE、MAPE

### 结果文件
- `results/static/`: 静态预测结果和可视化
- `results/dynamic/`: 动态预测结果和时间序列图
- `docs/实验记录.md`: 详细实验记录
- `docs/参数对比分析.md`: 参数敏感性分析

## 文档资源

- `docs/项目结构说明.md`: 详细的项目架构说明
- `docs/快速开始指南.md`: 详细的快速开始指南  
- `docs/实验记录.md`: 实验记录和结果分析
- `docs/参数对比分析.md`: 参数对比分析
- `api_service/README.md`: API服务详细文档
