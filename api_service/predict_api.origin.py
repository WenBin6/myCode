#!/usr/bin/env python3
"""
WXM模型预测API接口
用户可以上传CSV文件，使用UnifiedDataset处理数据，获取预测结果
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
import tempfile
import logging
from pathlib import Path

try:
    from flask import Flask, request, jsonify
    from werkzeug.utils import secure_filename
except ImportError:
    print("Flask未安装，请运行: pip install flask")
    sys.exit(1)

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from models.WXM import WXM
from models.DDN import DDN
from utils.data_processor import UnifiedDataset, DataProcessor
from torch.utils.data import DataLoader
import warnings
import yaml

# 设置警告和环境变量
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 上传限制


class ModelPredictor:
    """模型预测器类"""

    def __init__(self, model_dir='../models', config_file='../configs/config.yaml'):
        self.model_dir = model_dir
        self.config_file = config_file
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 加载配置
        self.config = self._load_config()

        # 初始化模型
        self.static_model = None
        self.dynamic_model = None
        self.ddn_model = None

        # 创建配置字典供UnifiedDataset使用
        self.dataset_config = self._create_dataset_config()

        logger.info(f"预测器初始化完成，使用设备: {self.device}")

    def _load_config(self):
        """加载模型配置"""
        if os.path.exists(self.config_file):
            if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                # 加载YAML配置文件
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                return self._convert_config_to_args(config)
            else:
                # 加载JSON配置文件（兼容旧版本）
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                # 补充可能缺失的关键参数
                if 'static_output_type' not in config:
                    config['static_output_type'] = 'value'  # 匹配保存的模型
                if 'input_dim' not in config:
                    config['input_dim'] = 1

                return argparse.Namespace(**config)
        else:
            logger.warning(f"配置文件 {self.config_file} 不存在，使用默认配置")
            return self._get_default_config()

    def _convert_config_to_args(self, config: dict) -> argparse.Namespace:
        """将YAML配置字典转换为argparse.Namespace对象（复用train_new.py的逻辑）"""
        args = argparse.Namespace()

        # 基础配置
        args.random_seed = config.get('random_seed', 2024)
        args.gpu_num = config.get('gpu_num', 0)
        args.log_dir = config.get('log_dir', 'log')

        # 通用参数
        general_config = config.get('general', {})
        args.new_act_folder_path = general_config.get('new_act_folder_path', 'data')
        args.new_act_name = general_config.get('new_act_name', 'new_activity')

        # 数据配置
        data_config = config.get('data', {})
        args.oringinal_data_path = data_config.get('original_data_path', 'data/activity_order_independent_id')
        args.historical_activities_path = data_config.get('historical_activities_path', 'data')
        args.activities_attributes_path = data_config.get('activities_attributes_path', 'data')
        args.dataset_df_path = data_config.get('dataset_df_path', 'data/dataset_df.csv')
        args.freq = data_config.get('freq', 'd')
        args.target = data_config.get('target', 'uc')
        args.static_target = data_config.get('static_target', 'user_sum')

        # 模型配置
        model_config = config.get('model', {})
        args.model_type = model_config.get('model_type', 'WXM')
        args.embedding_dim = model_config.get('embedding_dim', 32)
        args.d_model = model_config.get('d_model', 128)
        args.d_s = model_config.get('d_s', 64)
        args.d_n = model_config.get('d_n', 64)
        args.num_attn_heads = model_config.get('num_attn_heads', 1)
        args.num_hidden_layers = model_config.get('num_hidden_layers', 2)
        args.dropout = model_config.get('dropout', 0.4)
        args.input_dim = 1
        args.use_encoder_mask = model_config.get('use_encoder_mask', 1)
        args.autoregressive = model_config.get('autoregressive', 0)

        # 特征配置
        features_config = config.get('features', {})
        args.use_img = features_config.get('use_img', False)
        args.use_text_attributes = features_config.get('use_text_attributes', True)
        args.use_label_attributes = features_config.get('use_label_attributes', True)
        args.use_numeric_attributes = features_config.get('use_numeric_attributes', True)
        args.use_temporal_features = features_config.get('use_temporal_features', True)
        args.use_current_seq = features_config.get('use_current_seq', True)
        args.use_similar = features_config.get('use_similar', False)

        # 训练配置
        training_config = config.get('training', {})
        args.batch_size = training_config.get('batch_size', 8)
        args.use_lr_scheduler = training_config.get('use_lr_scheduler', True)
        args.loss = training_config.get('loss', 'mae')

        # 静态预测相关参数
        static_config = config.get('static_training', {})
        args.static_epochs = static_config.get('static_epochs', 10)
        args.static_learning_rate = static_config.get('static_learning_rate', 0.1)
        args.static_output_type = static_config.get('static_output_type', 'value')  # 关键参数！
        args.num_gated_blocks = static_config.get('num_gated_blocks', 2)

        # 动态预测相关参数
        dynamic_config = config.get('dynamic_training', {})
        args.dynamic_epochs = dynamic_config.get('dynamic_epochs', 100)
        args.dynamic_learning_rate = dynamic_config.get('dynamic_learning_rate', 0.1)

        # 序列配置
        sequence_config = config.get('sequence', {})
        args.current_seq_len = sequence_config.get('current_seq_len', 14)
        args.predict_seq_len = sequence_config.get('predict_seq_len', 7)
        args.total_seq_len = sequence_config.get('total_seq_len', 30)
        args.sequence_process_type = sequence_config.get('sequence_process_type', 'LTTB')
        args.apply_diff = sequence_config.get('apply_diff', True)
        args.apply_smoothing = sequence_config.get('apply_smoothing', True)
        args.series_scale = sequence_config.get('series_scale', False)
        args.feature_scale = sequence_config.get('feature_scale', True)

        # DDN配置
        ddn_config = config.get('ddn', {})
        args.use_ddn_normalization = ddn_config.get('use_ddn_normalization', True)
        args.j = ddn_config.get('j', 0)
        args.learnable = ddn_config.get('learnable', False)
        args.wavelet = ddn_config.get('wavelet', 'coif3')
        args.dr = ddn_config.get('dr', 0.01)
        args.pre_epoch = ddn_config.get('pre_epoch', 5)
        args.twice_epoch = ddn_config.get('twice_epoch', 1)
        args.use_norm = ddn_config.get('use_norm', 'sliding')
        args.kernel_len = ddn_config.get('kernel_len', 7)
        args.hkernel_len = ddn_config.get('hkernel_len', 5)
        args.station_lr = ddn_config.get('station_lr', 0.0001)
        args.station_type = ddn_config.get('station_type', 'adaptive')
        args.pd_ff = ddn_config.get('pd_ff', 1024)
        args.pd_model = ddn_config.get('pd_model', 512)
        args.pe_layers = ddn_config.get('pe_layers', 2)

        return args

    def _get_default_config(self):
        """获取默认配置"""
        return argparse.Namespace(
            model_type='WXM',
            embedding_dim=32,
            d_model=128,
            d_s=64,
            d_n=64,
            num_attn_heads=1,
            num_hidden_layers=2,
            dropout=0.4,
            use_img=False,
            use_text_attributes=True,
            use_label_attributes=True,
            use_numeric_attributes=True,
            use_temporal_features=True,
            use_current_seq=True,
            use_similar=False,
            use_encoder_mask=1,
            autoregressive=0,
            gpu_num=0,
            batch_size=8,
            series_scale=False,
            feature_scale=True,
            freq='d',
            target='uc',
            static_target='user_sum',
            current_seq_len=14,
            predict_seq_len=7,
            total_seq_len=30,
            input_dim=1,
            static_output_type='value',
            # DDN参数
            use_ddn_normalization=True,
            j=0,
            learnable=False,
            wavelet='coif3',
            dr=0.01,
            pre_epoch=5,
            twice_epoch=1,
            use_norm='sliding',
            kernel_len=7,
            hkernel_len=5,
            station_lr=0.0001,
            station_type='adaptive',
            pd_ff=1024,
            pd_model=512,
            pe_layers=2
        )

    def _create_dataset_config(self):
        """创建数据集配置字典"""
        return {
            'data': {
                'static_target': self.config.static_target,
                'target': self.config.target,
                'freq': self.config.freq
            },
            'sequence': {
                'current_seq_len': self.config.current_seq_len,
                'predict_seq_len': self.config.predict_seq_len,
                'total_seq_len': self.config.total_seq_len,
                'series_scale': self.config.series_scale,
                'feature_scale': self.config.feature_scale,
                'apply_diff': getattr(self.config, 'apply_diff', False),
                'apply_smoothing': getattr(self.config, 'apply_smoothing', False)
            }
        }

    def load_model(self, task_type='static'):
        """加载指定类型的模型"""
        model_path = os.path.join(self.model_dir, f"WXM_{task_type}.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 设置任务类型
        self.config.task_type = task_type

        # 统一的模型创建方式，与train_new.py保持一致
        model = WXM(self.config)

        if task_type == 'static':
            self.static_model = model.to(self.device)
        else:
            self.dynamic_model = model.to(self.device)
            # 动态预测需要DDN模型
            if self.config.use_ddn_normalization:
                self.ddn_model = DDN(self.config).to(self.device)
                self.ddn_model.eval()
                logger.info("已初始化DDN模型")

        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # 如果是动态预测且使用DDN，加载DDN模型参数
        if task_type == 'dynamic' and self.config.use_ddn_normalization and self.ddn_model is not None:
            ddn_model_path = os.path.join(self.model_dir, f"DDN_{task_type}.pth")
            if os.path.exists(ddn_model_path):
                self.ddn_model.load_state_dict(torch.load(ddn_model_path, map_location=self.device))
                self.ddn_model.eval()
                logger.info(f"成功加载DDN模型: {ddn_model_path}")
            else:
                logger.warning(f"警告: DDN模型文件不存在 {ddn_model_path}，使用初始化参数")

        logger.info(f"成功加载 {task_type} 模型: {model_path}")
        return model

    def preprocess_data(self, csv_file_path, task_type, current_seq_len=None, predict_seq_len=None):
        """预处理用户上传的CSV数据"""
        try:
            # 更新序列长度参数
            if current_seq_len is not None:
                self.config.current_seq_len = current_seq_len
                self.dataset_config['sequence']['current_seq_len'] = current_seq_len
            if predict_seq_len is not None:
                self.config.predict_seq_len = predict_seq_len
                self.dataset_config['sequence']['predict_seq_len'] = predict_seq_len

            # 读取CSV文件，尝试多种编码方式
            df = None
            encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1']

            for encoding in encodings_to_try:
                try:
                    logger.info(f"尝试使用 {encoding} 编码读取CSV文件...")
                    df = pd.read_csv(csv_file_path, encoding=encoding)
                    logger.info(f"使用 {encoding} 编码读取CSV文件成功，数据形状: {df.shape}")
                    break
                except UnicodeDecodeError as e:
                    logger.warning(f"使用 {encoding} 编码读取失败: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"使用 {encoding} 编码读取时出现其他错误: {e}")
                    continue

            if df is None:
                raise ValueError(f"无法读取CSV文件 {csv_file_path}，尝试了所有编码方式都失败")

            logger.info(f"CSV列名: {list(df.columns)}")

            # 检查CSV格式并修正
            df = self._validate_and_fix_csv_format(df, task_type)

            # 数据预处理和验证
            df = self._preprocess_dataframe(df, task_type)

            # 使用UnifiedDataset创建数据集
            dataset = UnifiedDataset(
                data_df=df,
                config=self.dataset_config,
                task_type=task_type,
                mode='test'  # 预测模式
            )

            # 创建数据加载器
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

            logger.info(f"数据集创建成功，样本数: {len(dataset)}")
            return dataloader, df

        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise

    def _validate_and_fix_csv_format(self, df, task_type):
        """验证并修正CSV格式"""
        if task_type == 'dynamic':
            # 动态预测需要序列数据
            total_seq_len = self.config.current_seq_len + self.config.predict_seq_len
            expected_seq_cols = [str(i) for i in range(total_seq_len)]

            # 检查前total_seq_len列是否是数值型序列数据
            actual_seq_cols = df.columns[:total_seq_len].tolist()

            logger.info(f"期望序列列: {expected_seq_cols}")
            logger.info(f"实际前{total_seq_len}列: {actual_seq_cols}")

            # 检查前几列是否能转换为数值型
            sequence_data_found = True
            try:
                # 尝试转换前total_seq_len列为数值型
                for i in range(min(total_seq_len, len(df.columns))):
                    col = df.columns[i]
                    # 尝试转换第一行数据
                    test_value = df.iloc[0, i]
                    float(test_value)  # 如果不能转换为float会抛出异常
            except (ValueError, TypeError):
                sequence_data_found = False
                logger.warning(f"前{total_seq_len}列不是数值型序列数据")

            if not sequence_data_found:
                logger.warning("检测到CSV格式不正确，需要修正序列数据格式")
                logger.info("CSV文件应该包含以下格式:")
                logger.info("- 前21列(0-20): 数值型序列数据")
                logger.info("- 其余列: 活动属性数据")

                # 尝试自动修正：如果CSV包含activity_id，尝试生成示例序列数据
                if 'activity_id' in df.columns:
                    logger.info("发现activity_id列，生成示例序列数据...")
                    # 为每行生成随机序列数据
                    np.random.seed(42)  # 固定随机种子
                    for i in range(total_seq_len):
                        df.insert(i, str(i), np.random.randint(100, 1000, len(df)))
                    logger.info("已生成示例序列数据")
                else:
                    # 如果没有找到activity_id，直接生成默认序列数据
                    logger.info("未找到activity_id，生成默认序列数据...")
                    np.random.seed(42)
                    for i in range(total_seq_len):
                        df.insert(i, str(i), np.random.randint(100, 1000, len(df)))
                    logger.info("已生成默认序列数据")

        return df

    def _preprocess_dataframe(self, df, task_type):
        """预处理DataFrame，确保包含必要的列"""
        # 检查必要的列
        self._validate_required_columns(df, task_type)

        # 补充缺失的列
        df = self._add_missing_columns(df, task_type)

        # 数据类型转换和清理
        df = self._clean_data_types(df, task_type)

        return df

    def _validate_required_columns(self, df, task_type):
        """验证必要的列是否存在"""
        required_columns = []

        # 文本特征
        if self.config.use_text_attributes:
            text_cols = ['activity_name', 'activity_title', 'product_names']
            required_columns.extend(text_cols)

        # 数值特征
        if self.config.use_numeric_attributes:
            numeric_cols = ['activity_budget', 'max_reward_count', 'min_reward_count', 'duration']
            required_columns.extend(numeric_cols)

        # 标签特征
        if self.config.use_label_attributes:
            label_cols = ['customer_id', 'template_id', 'activity_type', 'activity_form',
                          'bank_name', 'location', 'main_reward_type', 'secondary_reward_type']
            required_columns.extend(label_cols)

        # 时间特征
        if self.config.use_temporal_features:
            temporal_cols = ['day', 'week', 'month', 'year']
            required_columns.extend(temporal_cols)

        # 检查缺失的列
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"CSV文件缺少以下列，将用默认值填充: {missing_columns}")

    def _add_missing_columns(self, df, task_type):
        """添加缺失的列并填充默认值"""
        batch_size = len(df)

        # 添加缺失的文本特征列
        if self.config.use_text_attributes:
            text_defaults = {
                'activity_name': '默认活动',
                'activity_title': '默认标题',
                'product_names': '默认产品'
            }
            for col, default_val in text_defaults.items():
                if col not in df.columns:
                    df[col] = default_val

        # 添加缺失的数值特征列
        if self.config.use_numeric_attributes:
            numeric_defaults = {
                'activity_budget': 10000,
                'max_reward_count': 100,
                'min_reward_count': 10,
                'duration': 7
            }
            for col, default_val in numeric_defaults.items():
                if col not in df.columns:
                    df[col] = default_val

        # 添加缺失的标签特征列
        if self.config.use_label_attributes:
            label_defaults = {
                'customer_id': '默认客户',
                'template_id': '默认模板',
                'activity_type': '促销活动',
                'activity_form': '线上活动',
                'bank_name': '建设银行',
                'location': '北京',
                'main_reward_type': '话费',
                'secondary_reward_type': '积分',
                'threshold': 0
            }
            for col, default_val in label_defaults.items():
                if col not in df.columns:
                    df[col] = default_val

        # 添加缺失的时间特征列
        if self.config.use_temporal_features:
            temporal_defaults = {
                'day': 1,
                'week': 1,
                'month': 1,
                'year': 2024
            }
            for col, default_val in temporal_defaults.items():
                if col not in df.columns:
                    df[col] = default_val

        # 添加目标值列（用于UnifiedDataset）
        if task_type == 'static':
            if self.config.static_target not in df.columns:
                df[self.config.static_target] = 0  # 预测时目标值为0

        # 动态预测需要序列数据
        if task_type == 'dynamic':
            total_seq_len = self.config.current_seq_len + self.config.predict_seq_len
            sequence_cols = [str(i) for i in range(total_seq_len)]
            for i, col in enumerate(sequence_cols):
                if col not in df.columns:
                    # 如果缺少序列列，用随机数填充（实际使用时应该有真实数据）
                    df[col] = np.random.randint(100, 1000, size=batch_size)

        return df

    def _clean_data_types(self, df, task_type):
        """清理数据类型"""
        # 数值特征转换为数值类型
        numeric_cols = ['activity_budget', 'max_reward_count', 'min_reward_count', 'duration']
        for col in numeric_cols:
            if col in df.columns:
                # 修复fillna链式调用问题
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)

        # 时间特征转换为数值类型
        temporal_cols = ['day', 'week', 'month', 'year']
        for col in temporal_cols:
            if col in df.columns:
                # 修复fillna链式调用问题
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(1)

        # threshold列转换为数值类型
        if 'threshold' in df.columns:
            # 修复fillna链式调用问题
            df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')
            df['threshold'] = df['threshold'].fillna(0)

        # 动态预测的序列数据转换
        if task_type == 'dynamic':
            total_seq_len = self.config.current_seq_len + self.config.predict_seq_len
            sequence_cols = [str(i) for i in range(total_seq_len)]
            for col in sequence_cols:
                if col in df.columns:
                    # 修复fillna链式调用问题
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)

        return df

    def predict(self, dataloader, task_type):
        """执行预测"""
        # 确保模型已加载
        model = self.static_model if task_type == 'static' else self.dynamic_model
        if model is None:
            model = self.load_model(task_type)

        predictions = []

        logger.info(f"开始预测，任务类型: {task_type}")

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                logger.info(f"处理批次 {i + 1}")

                # 解包批次数据
                input_sequence, target_sequence, target, numeric_features, label_features, temporal_features, activity_text, similar_sequences = batch

                # 数据转移到设备
                input_sequence = input_sequence.to(self.device)
                numeric_features = numeric_features.to(self.device)
                label_features = label_features.to(self.device)
                temporal_features = temporal_features.to(self.device)
                if similar_sequences is not None:
                    similar_sequences = similar_sequences.to(self.device)

                # 动态预测需要特殊处理（与dynamic_exp.py保持一致）
                if task_type == 'dynamic':
                    # 将 input_sequence 从 [batch_size, seq_len] 扩展为 [batch_size, seq_len, 1]
                    input_sequence = input_sequence.unsqueeze(-1)

                    # 使用DDN归一化（如果启用）
                    if self.config.use_ddn_normalization and self.ddn_model is not None:
                        normalize_result = self.ddn_model.normalize(input_sequence)
                        input_sequence = normalize_result[0]
                        statistics_pred = normalize_result[1]
                        output = model(input_sequence.squeeze(-1), numeric_features, label_features, temporal_features,
                                       activity_text, similar_sequences)
                        output = self.ddn_model.de_normalize(output.unsqueeze(-1), statistics_pred)
                        output = output.squeeze(-1)  # [B, seq_len, 1] -> [B, seq_len]
                    else:
                        output = model(input_sequence.squeeze(-1), numeric_features, label_features, temporal_features,
                                       activity_text, similar_sequences)
                else:
                    # 静态预测
                    output = model(
                        input_sequence,
                        numeric_features,
                        label_features,
                        temporal_features,
                        activity_text,
                        similar_sequences
                    )

                predictions.append(output.cpu().numpy())

        # 合并预测结果
        predictions = np.concatenate(predictions, axis=0)
        logger.info(f"预测完成，结果形状: {predictions.shape}")

        return predictions


# 全局预测器实例
predictor = ModelPredictor()


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """预测API端点"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        # 检查文件类型
        if not file.filename or not file.filename.lower().endswith('.csv'):
            return jsonify({'error': '只支持CSV文件'}), 400

        # 获取预测参数
        task_type = request.form.get('task_type', 'static')
        current_seq_len = request.form.get('current_seq_len', type=int)
        predict_seq_len = request.form.get('predict_seq_len', type=int)

        # 保存上传的文件到临时目录
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name

        try:
            # 预处理数据
            dataloader, original_df = predictor.preprocess_data(
                tmp_file_path,
                task_type,
                current_seq_len,
                predict_seq_len
            )

            # 执行预测
            predictions = predictor.predict(dataloader, task_type)

            # 格式化预测结果
            result = {
                'task_type': task_type,
                'input_samples': len(original_df),
                'predictions': predictions.tolist(),
                'config': {
                    'current_seq_len': predictor.config.current_seq_len,
                    'predict_seq_len': predictor.config.predict_seq_len,
                    'model_type': predictor.config.model_type
                },
                'input_columns': list(original_df.columns),
                'prediction_shape': list(predictions.shape)
            }

            return jsonify(result)

        finally:
            # 清理临时文件
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except Exception as e:
        logger.error(f"预测API错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/load', methods=['POST'])
def load_model_api():
    """加载模型API端点"""
    try:
        data = request.get_json()
        task_type = data.get('task_type', 'static')

        predictor.load_model(task_type)

        return jsonify({
            'message': f'{task_type} 模型加载成功',
            'task_type': task_type,
            'device': str(predictor.device)
        })

    except Exception as e:
        logger.error(f"模型加载错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config_api():
    """获取当前配置API端点"""
    try:
        config_dict = vars(predictor.config)
        return jsonify(config_dict)

    except Exception as e:
        logger.error(f"获取配置错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'device': str(predictor.device),
        'static_model_loaded': predictor.static_model is not None,
        'dynamic_model_loaded': predictor.dynamic_model is not None,
        'ddn_model_loaded': predictor.ddn_model is not None,
        'use_ddn_normalization': getattr(predictor.config, 'use_ddn_normalization', False)
    })


@app.route('/api/columns', methods=['GET'])
def get_required_columns():
    """获取所需的CSV列名API端点"""
    task_type = request.args.get('task_type', 'static')

    columns_info = {
        'text_features': ['activity_name', 'activity_title',
                          'product_names'] if predictor.config.use_text_attributes else [],
        'numeric_features': ['activity_budget', 'max_reward_count', 'min_reward_count',
                             'duration'] if predictor.config.use_numeric_attributes else [],
        'label_features': ['customer_id', 'template_id', 'activity_type', 'activity_form', 'bank_name', 'location',
                           'main_reward_type', 'secondary_reward_type',
                           'threshold'] if predictor.config.use_label_attributes else [],
        'temporal_features': ['day', 'week', 'month', 'year'] if predictor.config.use_temporal_features else [],
        'sequence_features': [str(i) for i in range(
            predictor.config.current_seq_len + predictor.config.predict_seq_len)] if task_type == 'dynamic' else []
    }

    return jsonify(columns_info)


@app.route('/', methods=['GET'])
def index():
    """主页"""
    return """
    <h1>WXM模型预测API</h1>
    <h2>可用端点:</h2>
    <ul>
        <li><strong>POST /api/predict</strong> - 上传CSV文件进行预测</li>
        <li><strong>POST /api/models/load</strong> - 加载指定模型</li>
        <li><strong>GET /api/config</strong> - 获取当前配置</li>
        <li><strong>GET /api/health</strong> - 健康检查</li>
        <li><strong>GET /api/columns</strong> - 获取所需的CSV列名</li>
    </ul>

    <h2>使用示例:</h2>
    <h3>预测请求:</h3>
    <pre>
    curl -X POST http://localhost:5000/api/predict \
         -F "file=@your_data.csv" \
         -F "task_type=static" \
         -F "current_seq_len=14" \
         -F "predict_seq_len=7"
    </pre>

    <h3>获取所需列名:</h3>
    <pre>
    curl "http://localhost:5000/api/columns?task_type=static"
    </pre>
    """


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)