# WXM模型预测API接口

## 概述

这是一个基于WXM模型的预测API接口，支持静态预测和动态预测两种模式。用户可以通过上传CSV文件来获取预测结果。

## 预测模式

### 1. 静态预测 (task_type=static)

- **用途**: 基于活动属性特征进行单点预测或分位数预测
- **输入**: 活动属性特征（文本、数值、标签、时间特征）
- **输出**: 单点预测值或分位数预测
- **适用场景**: 活动效果评估、用户参与度预测等

### 2. 动态预测 (task_type=dynamic)

- **用途**: 基于历史序列数据预测未来时间序列
- **输入**: 14天的历史序列数据 + 活动属性特征
- **输出**: 7天的预测序列数据
- **适用场景**: 活动参与趋势预测、用户行为序列预测等

## API端点

### 主要端点

1. **POST /api/predict** - 上传CSV文件进行预测
2. **POST /api/models/load** - 加载指定模型
3. **GET /api/config** - 获取当前配置
4. **GET /api/health** - 健康检查
5. **GET /api/columns** - 获取所需的CSV列名

## CSV文件格式要求

### 静态预测CSV格式

```
必需列:
- activity_name: 活动名称
- activity_title: 活动标题  
- product_names: 产品名称
- activity_budget: 活动预算
- max_reward_count: 最大奖励数
- min_reward_count: 最小奖励数
- duration: 持续时间
- customer_id: 客户ID
- template_id: 模板ID
- activity_type: 活动类型
- activity_form: 活动形式
- bank_name: 银行名称
- location: 位置
- main_reward_type: 主要奖励类型
- secondary_reward_type: 次要奖励类型
- day, week, month, year: 时间特征
- user_sum: 目标值列 (预测时填充0)
```

### 动态预测CSV格式

```
必需列:
- 前14列 (0-13): 数值型序列数据 [batch_size, 14]
  - 列名必须严格为: "0", "1", "2", ..., "13"
  - 列0-13: 14天的历史序列数据 (输入)
  - 所有序列数据必须为数值型，不允许缺失值
- 其余列: 同上静态预测的所有特征列

注意事项:
- 系统会严格验证前14列的数据格式
- 列名必须完全匹配，不允许自动修正
- 序列数据必须完整，系统不会生成随机数据填充
```

## 使用示例

### 动态预测请求

```bash
curl -X POST http://localhost:5000/api/predict \
     -F "file=@your_data.csv" \
     -F "task_type=dynamic" \
     -F "current_seq_len=14" \
     -F "predict_seq_len=7"
```

### 静态预测请求

```bash
curl -X POST http://localhost:5000/api/predict \
     -F "file=@your_data.csv" \
     -F "task_type=static"
```

### 获取所需列名

```bash
curl "http://localhost:5000/api/columns?task_type=dynamic"
```

## 输入输出维度说明

### 动态预测维度

- **输入序列**: `[batch_size, 14]` - 14天的历史数据
- **输出序列**: `[batch_size, 7]` - 7天的预测数据
- **输入序列长度**: 14天

### 静态预测维度

- **输入**: 活动属性特征向量
- **输出**: `[batch_size, 1]` (单点预测) 或 `[batch_size, num_quantiles]` (分位数预测)

## 模型配置

### 默认配置参数

```python
current_seq_len = 14      # 当前序列长度
predict_seq_len = 7       # 预测序列长度  
total_seq_len = 30        # 总序列长度
input_dim = 1             # 序列输入维度
d_model = 128             # 模型隐藏维度
d_s = 64                  # 静态预测隐藏维度
d_n = 64                  # 动态预测隐藏维度
batch_size = 8            # 批次大小
```

## 响应格式

### 预测响应

```json
{
    "task_type": "dynamic",
    "input_samples": 100,
    "predictions": [[...], [...], ...],
    "config": {
        "current_seq_len": 14,
        "predict_seq_len": 7,
        "model_type": "WXM"
    },
    "input_columns": ["列名列表"],
    "prediction_shape": [100, 7],
    "dimension_info": {
        "input_sequence_length": 14,
        "output_sequence_length": 7,
        "description": "动态预测：输入14天，输出7天"
    }
}
```

### 列名信息响应

```json
{
    "text_features": ["activity_name", "activity_title", "product_names"],
    "numeric_features": ["activity_budget", "max_reward_count", "min_reward_count", "duration"],
    "label_features": ["customer_id", "template_id", "activity_type", "activity_form", "bank_name", "location", "main_reward_type", "secondary_reward_type", "threshold"],
    "temporal_features": ["day", "week", "month", "year"],
    "sequence_features": ["0", "1", "2", ..., "13"],
    "dimension_info": {
        "input_sequence_length": 14,
        "output_sequence_length": 7,
        "description": "动态预测模式：输入14天历史数据，输出7天预测数据"
    }
}
```

## 注意事项

1. **序列数据要求**: 动态预测的前14列必须是数值型数据，列名必须严格为"0","1","2",...,"13"
2. **严格验证**: 系统会严格验证序列数据格式，不允许自动修正或生成随机数据
3. **特征完整性**: 如果CSV缺少某些特征列，API会自动填充默认值（仅限活动属性特征）
4. **数据编码**: 支持UTF-8、GBK、GB2312、Latin1等编码格式
5. **文件大小**: 上传文件大小限制为100MB
6. **模型加载**: 首次使用前需要调用 `/api/models/load`端点加载模型

## 错误处理

API会返回详细的错误信息，包括：

- 文件格式错误
- 数据预处理失败
- 模型加载失败
- 预测执行错误

### 常见错误示例

#### 动态预测序列数据错误
```json
{
    "error": "第1列列名应为'0'，实际为'day'"
}
```

#### 序列列缺失错误
```json
{
    "error": "动态预测缺少必要的序列列: ['0', '1', '2']。请确保CSV文件前14列(0-13)包含完整的序列数据"
}
```

#### 非数值型数据错误
```json
{
    "error": "第5列(列名'4')包含非数值型数据，无法转换为数值类型: could not convert string to float: 'abc'"
}
```

#### 列数不足错误
```json
{
    "error": "CSV文件列数不足，需要至少14列序列数据，当前只有10列"
}
```

## 启动服务

```bash
cd api_service
python predict_api.py
```

服务将在 `http://localhost:5000` 启动。
