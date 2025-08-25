#!/usr/bin/env python3
"""
WXM模型动态预测API接口
基于原有的数据处理方式和配置，提供动态预测服务
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Flask未安装，请运行: pip install flask")
    sys.exit(1)

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from models.WXM import WXM
from models.DDN import DDN
from utils.data_processor import DataProcessor, UnifiedDataset
from torch.utils.data import DataLoader
import warnings

# 设置警告和环境变量
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 上传限制


class ModelPredictor:
    """模型预测器类，支持静态和动态预测"""
    
    def __init__(self, config_file: str = 'configs/config.yaml'):
        self.config_file = config_file
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        self.config = self._load_config()
        logger.info(f"配置加载完成，使用设备: {self.device}")
        
        # 初始化模型
        self.static_model = None
        self.dynamic_model = None
        self.ddn_model = None
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(self.config)
        
        # 创建数据集配置
        self.dataset_config = self._create_dataset_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("配置文件加载成功")
        return config
    
    def load_models(self, task_type='dynamic'):
        """加载指定类型的模型"""
        try:
            if task_type == 'static':
                # 加载静态预测模型
                wxm_model_path = 'models/WXM_static.pth'
                if not os.path.exists(wxm_model_path):
                    raise FileNotFoundError(f"静态WXM模型文件不存在: {wxm_model_path}")
                
                model_args = self._create_model_args('static')
                self.static_model = WXM(model_args)
                
                # 加载静态模型权重
                checkpoint = torch.load(wxm_model_path, map_location=self.device)
                self.static_model.load_state_dict(checkpoint, strict=False)
                self.static_model.to(self.device)
                self.static_model.eval()
                logger.info(f"静态WXM模型加载成功: {wxm_model_path}")
            else:
                # 加载动态预测模型
                wxm_model_path = 'models/WXM_dynamic.pth'
                if not os.path.exists(wxm_model_path):
                    raise FileNotFoundError(f"动态WXM模型文件不存在: {wxm_model_path}")
                
                model_args = self._create_model_args('dynamic')
                self.dynamic_model = WXM(model_args)
                
                # 加载模型权重，只加载动态预测相关的部分
                checkpoint = torch.load(wxm_model_path, map_location=self.device)
                dynamic_weights = {}
                for key, value in checkpoint.items():
                    if not key.startswith('static_predictor'):
                        dynamic_weights[key] = value
                
                # 加载过滤后的权重
                missing_keys, unexpected_keys = self.dynamic_model.load_state_dict(dynamic_weights, strict=False)
                if missing_keys:
                    logger.warning(f"缺失的权重键: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"意外的权重键: {unexpected_keys}")
                
                self.dynamic_model.to(self.device)
                self.dynamic_model.eval()
                logger.info(f"动态WXM模型加载成功: {wxm_model_path}")
                
                # 加载DDN模型（如果启用）
                if self.config['ddn']['use_ddn_normalization']:
                    ddn_model_path = 'models/DDN_dynamic.pth'
                    if os.path.exists(ddn_model_path):
                        self.ddn_model = DDN(model_args)
                        self.ddn_model.load_state_dict(torch.load(ddn_model_path, map_location=self.device))
                        self.ddn_model.to(self.device)
                        self.ddn_model.eval()
                        logger.info(f"DDN模型加载成功: {ddn_model_path}")
                    else:
                        logger.warning(f"DDN模型文件不存在: {ddn_model_path}，将使用初始化参数")
                        self.ddn_model = DDN(model_args).to(self.device)
                        self.ddn_model.eval()
                else:
                    logger.info("未启用DDN归一化")
            
            logger.info(f"{task_type}模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def _create_model_args(self, task_type='dynamic'):
        """创建模型参数对象"""
        class ModelArgs:
            def __init__(self, config, task_type):
                # 基础配置
                self.model_type = config['model']['model_type']
                self.embedding_dim = config['model']['embedding_dim']
                self.d_model = config['model']['d_model']
                self.d_s = config['model']['d_s']  # 静态预测使用d_s
                self.d_n = config['model']['d_n']  # 动态预测使用d_n
                self.num_attn_heads = config['model']['num_attn_heads']
                self.num_hidden_layers = config['model']['num_hidden_layers']
                self.dropout = config['model']['dropout']
                self.input_dim = 1
                self.use_encoder_mask = config['model']['use_encoder_mask']
                self.autoregressive = config['model']['autoregressive']
                
                # 特征配置
                self.use_img = config['features']['use_img']
                self.use_text_attributes = config['features']['use_text_attributes']
                self.use_label_attributes = config['features']['use_label_attributes']
                self.use_numeric_attributes = config['features']['use_numeric_attributes']
                self.use_temporal_features = config['features']['use_temporal_features']
                self.use_current_seq = config['features']['use_current_seq']
                self.use_similar = config['features']['use_similar']
        
        # 序列配置
                self.current_seq_len = config['sequence']['current_seq_len']
                self.predict_seq_len = config['sequence']['predict_seq_len']
                self.total_seq_len = config['sequence']['total_seq_len']
                
                # 任务类型
                self.task_type = task_type
                
                # GPU配置
                self.gpu_num = config.get('gpu_num', 0)
                
                # 静态预测相关参数
                self.static_output_type = config.get('static_training', {}).get('static_output_type', 'value')
                self.num_gated_blocks = config.get('static_training', {}).get('num_gated_blocks', 2)
                
                # DDN配置
                self.use_ddn_normalization = config['ddn']['use_ddn_normalization']
                self.j = config['ddn']['j']
                self.learnable = config['ddn']['learnable']
                self.wavelet = config['ddn']['wavelet']
                self.dr = config['ddn']['dr']
                self.pre_epoch = config['ddn']['pre_epoch']
                self.twice_epoch = config['ddn']['twice_epoch']
                self.use_norm = config['ddn']['use_norm']
                self.kernel_len = config['ddn']['kernel_len']
                self.hkernel_len = config['ddn']['hkernel_len']
                self.station_lr = config['ddn']['station_lr']
                self.station_type = config['ddn']['station_type']
                self.pd_ff = config['ddn']['pd_ff']
                self.pd_model = config['ddn']['pd_model']
                self.pe_layers = config['ddn']['pe_layers']
        
        return ModelArgs(self.config, task_type)
    
    def preprocess_data(self, csv_file_path: str, task_type='dynamic') -> pd.DataFrame:
        """预处理CSV数据，使用原有的数据处理方式"""
        try:
            logger.info(f"=== 开始{task_type}数据预处理 ===")
            
            # 读取CSV文件
            df = self._read_csv_file(csv_file_path)
            logger.info(f"CSV文件读取成功，数据形状: {df.shape}")
            logger.info(f"列名: {list(df.columns)}")
            
            # 验证数据格式
            self._validate_data_format(df, task_type)
            logger.info("数据格式验证通过")
            
            if task_type == 'dynamic':
                # 输出原始序列数据的详细信息
                logger.info("=== CSV文件中的原始序列数据 ===")
                current_seq_len = self.config['sequence']['current_seq_len']
                for i in range(min(3, len(df))):  # 显示前3行数据
                    logger.info(f"第{i+1}行数据:")
                    for j in range(current_seq_len):
                        col = str(j)
                        if col in df.columns:
                            val = df.iloc[i][col]
                            logger.info(f"  列{col}(第{j+1}天): {val}")
                    logger.info("  ---")
                logger.info("=== 原始序列数据检查完成 ===")
            
            # 数据预处理
            processed_df = self._process_dataframe(df)
            logger.info(f"数据预处理完成，最终形状: {processed_df.shape}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def _read_csv_file(self, file_path: str) -> pd.DataFrame:
        """读取CSV文件，支持多种编码"""
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"尝试使用 {encoding} 编码读取文件...")
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"使用 {encoding} 编码读取成功")
                return df
            except UnicodeDecodeError:
                logger.warning(f"使用 {encoding} 编码读取失败")
                continue
            except Exception as e:
                logger.warning(f"使用 {encoding} 编码读取时出现其他错误: {e}")
                continue
        
        raise ValueError(f"无法读取CSV文件，尝试了所有编码方式都失败")
    
    def _validate_data_format(self, df: pd.DataFrame, task_type='dynamic'):
        """验证数据格式"""
        if task_type == 'static':
            # 静态预测不需要验证序列数据
            logger.info("静态预测模式，跳过序列数据验证")
            return
        else:
            # 动态预测需要验证序列数据
            current_seq_len = self.config['sequence']['current_seq_len']
            expected_seq_cols = [str(i) for i in range(current_seq_len)]
            
            if len(df.columns) < current_seq_len:
                raise ValueError(f"CSV文件列数不足，需要至少{current_seq_len}列序列数据")
            
            # 验证前current_seq_len列是否为数值型
            for i in range(current_seq_len):
                col = df.columns[i]
                if col != str(i):
                    raise ValueError(f"第{i+1}列列名应为'{i}'，实际为'{col}'")
                
                try:
                    pd.to_numeric(df.iloc[:, i], errors='raise')
                except (ValueError, TypeError) as e:
                    raise ValueError(f"第{i+1}列包含非数值型数据: {e}")
            
            logger.info(f"序列数据验证通过，前{current_seq_len}列均为有效的数值型数据")
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理DataFrame，补充缺失列"""
        # 补充必要的特征列
        self._add_missing_columns(df)
        
        # 数据类型转换
        self._convert_data_types(df)
        
        return df
    
    def _add_missing_columns(self, df: pd.DataFrame):
        """添加缺失的列并填充默认值"""
        # 文本特征
        text_defaults = {
            'activity_name': '默认活动',
            'activity_title': '默认标题',
            'product_names': '默认产品'
        }
        for col, default_val in text_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            logger.info(f"添加缺失列: {col} = {default_val}")
        
        # 数值特征
        numeric_defaults = {
            'activity_budget': 10000,
            'max_reward_count': 100,
            'min_reward_count': 10,
            'duration': 7
        }
        for col, default_val in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            logger.info(f"添加缺失列: {col} = {default_val}")
        
        # 标签特征
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
            logger.info(f"添加缺失列: {col} = {default_val}")
        
        # 添加编码列（UnifiedDataset需要）
        label_cols = ['customer_id', 'template_id', 'activity_type', 'activity_form', 
                     'bank_name', 'location', 'main_reward_type', 'secondary_reward_type']
        for col in label_cols:
            if col in df.columns:
                encoded_col = f'{col}_encoded'
                if encoded_col not in df.columns:
                    # 简单的数值编码
                    df[encoded_col] = df[col].astype('category').cat.codes
                    logger.info(f"添加编码列: {encoded_col}")
        
        # 时间特征
            temporal_defaults = {
                'day': 1,
                'week': 1,
                'month': 1,
                'year': 2024
            }
            for col, default_val in temporal_defaults.items():
                if col not in df.columns:
                    df[col] = default_val
                logger.info(f"添加缺失列: {col} = {default_val}")
    
    def _convert_data_types(self, df: pd.DataFrame):
        """转换数据类型"""
        # 数值特征
        numeric_cols = ['activity_budget', 'max_reward_count', 'min_reward_count', 'duration']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 时间特征
        temporal_cols = ['day', 'week', 'month', 'year']
        for col in temporal_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1)
        
        # threshold列
        if 'threshold' in df.columns:
            df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce').fillna(0)
        
        # 序列数据
        current_seq_len = self.config['sequence']['current_seq_len']
        sequence_cols = [str(i) for i in range(current_seq_len)]
        for col in sequence_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        logger.info("数据类型转换完成")
        
        # 输出转换后的序列数据
        logger.info("=== 数据类型转换后的序列数据 ===")
        for i in range(min(3, len(df))):  # 显示前3行数据
            logger.info(f"第{i+1}行转换后的序列数据:")
            for j in range(current_seq_len):
                col = str(j)
                if col in df.columns:
                    val = df.iloc[i][col]
                    logger.info(f"  列{col}(第{j+1}天): {val:.6f} (类型: {type(val)})")
            logger.info("  ---")
        logger.info("=== 转换后序列数据检查完成 ===")
    
    def _create_dynamic_dataset(self, df: pd.DataFrame):
        """创建动态预测数据集，确保序列数据正确传递"""
        class DynamicDataset:
            def __init__(self, data_df, config, predictor):
                self.data_df = data_df
                self.config = config
                self.predictor = predictor
                self.current_seq_len = config['sequence']['current_seq_len']
                self.predict_seq_len = config['sequence']['predict_seq_len']
                self.total_seq_len = config['sequence']['total_seq_len']
                
                logger.info(f"创建动态数据集，序列长度: {self.current_seq_len}")
            
            def __len__(self):
                return len(self.data_df)
            
            def __getitem__(self, idx):
                row = self.data_df.iloc[idx]
                
                # 提取序列数据 - 确保前14列是数值型
                input_values = []
                for i in range(self.current_seq_len):
                    col = str(i)
                    if col in row:
                        val = float(row[col])
                        input_values.append(val)
                    else:
                        input_values.append(0.0)
                
                input_sequence = torch.tensor(input_values, dtype=torch.float32)
                
                # 创建目标序列（预测时不需要，用零填充）
                target_sequence = torch.zeros(self.predict_seq_len, dtype=torch.float32)
                
                # 提取特征
                numeric_features = torch.tensor([
                    float(row.get('activity_budget', 10000)),
                    float(row.get('max_reward_count', 100)),
                    float(row.get('min_reward_count', 10)),
                    float(row.get('duration', 7))
                ], dtype=torch.float32)
                
                # 标签特征编码
                label_features = torch.tensor([
                    float(row.get('customer_id_encoded', 0)),
                    float(row.get('activity_type_encoded', 0)),
                    float(row.get('activity_form_encoded', 0)),
                    float(row.get('bank_name_encoded', 0)),
                    float(row.get('location_encoded', 0)),
                    float(row.get('main_reward_type_encoded', 0)),
                    float(row.get('secondary_reward_type_encoded', 0)),
                    float(row.get('template_id_encoded', 0)),
                    float(row.get('threshold', 0))
                ], dtype=torch.float32)
                
                # 时间特征
                temporal_features = torch.tensor([
                    float(row.get('day', 1)),
                    float(row.get('week', 1)),
                    float(row.get('month', 1)),
                    float(row.get('year', 2024))
                ], dtype=torch.float32)
                
                # 文本特征
                activity_text = {
                    'activity_name': str(row.get('activity_name', 'default')),
                    'activity_title': str(row.get('activity_title', 'default')),
                    'product_names': str(row.get('product_names', 'default'))
                }
                
                # 相似序列（用零填充）
                similar_sequences = torch.zeros(self.total_seq_len, dtype=torch.float32)
                
                # 目标值（预测时不需要）
                target = torch.tensor(0.0, dtype=torch.float32)
                
                logger.info(f"数据集样本 {idx} 序列数据: {input_values[:5]}... (前5个值)")
                
                return (input_sequence, target_sequence, target, numeric_features, 
                        label_features, temporal_features, activity_text, similar_sequences)
        
        return DynamicDataset(df, self.dataset_config, self)
    
    def _create_static_dataset(self, df: pd.DataFrame):
        """创建静态预测数据集"""
        class StaticDataset:
            def __init__(self, data_df, config, predictor):
                self.data_df = data_df
                self.config = config
                self.predictor = predictor
                
                logger.info(f"创建静态数据集，样本数: {len(data_df)}")
            
            def __len__(self):
                return len(self.data_df)
            
            def __getitem__(self, idx):
                row = self.data_df.iloc[idx]
                
                # 静态预测不需要序列数据
                input_sequence = torch.tensor([], dtype=torch.float32)
                target_sequence = torch.tensor([], dtype=torch.float32)
                
                # 提取特征
                numeric_features = torch.tensor([
                    float(row.get('activity_budget', 10000)),
                    float(row.get('max_reward_count', 100)),
                    float(row.get('min_reward_count', 10)),
                    float(row.get('duration', 7))
                ], dtype=torch.float32)
                
                # 标签特征编码
                label_features = torch.tensor([
                    float(row.get('customer_id_encoded', 0)),
                    float(row.get('activity_type_encoded', 0)),
                    float(row.get('activity_form_encoded', 0)),
                    float(row.get('bank_name_encoded', 0)),
                    float(row.get('location_encoded', 0)),
                    float(row.get('main_reward_type_encoded', 0)),
                    float(row.get('secondary_reward_type_encoded', 0)),
                    float(row.get('template_id_encoded', 0)),
                    float(row.get('threshold', 0))
                ], dtype=torch.float32)
                
                # 时间特征
                temporal_features = torch.tensor([
                    float(row.get('day', 1)),
                    float(row.get('week', 1)),
                    float(row.get('month', 1)),
                    float(row.get('year', 2024))
                ], dtype=torch.float32)
                
                # 文本特征
                activity_text = {
                    'activity_name': str(row.get('activity_name', 'default')),
                    'activity_title': str(row.get('activity_title', 'default')),
                    'product_names': str(row.get('product_names', 'default'))
                }
                
                # 相似序列（静态预测不需要）
                similar_sequences = torch.tensor([], dtype=torch.float32)
                
                # 目标值（预测时不需要）
                target = torch.tensor(0.0, dtype=torch.float32)
                
                logger.info(f"静态数据集样本 {idx} 特征提取完成")
                
                return (input_sequence, target_sequence, target, numeric_features, 
                        label_features, temporal_features, activity_text, similar_sequences)
        
        return StaticDataset(df, self.dataset_config, self)
    
    def _create_dataset_config(self):
        """创建数据集配置字典，与UnifiedDataset期望的格式保持一致"""
        return {
            'data': {
                'static_target': self.config['prediction']['static_target'],
                'target': self.config['prediction']['target'],
                'freq': self.config['prediction']['freq']
            },
            'sequence': {
                'current_seq_len': self.config['sequence']['current_seq_len'],
                'predict_seq_len': self.config['sequence']['predict_seq_len'],
                'total_seq_len': self.config['sequence']['total_seq_len'],
                'series_scale': self.config['sequence']['series_scale'],
                'feature_scale': self.config['sequence']['feature_scale']
            },
            'features': {
                'use_img': self.config['features']['use_img'],
                'use_text_attributes': self.config['features']['use_text_attributes'],
                'use_label_attributes': self.config['features']['use_label_attributes'],
                'use_numeric_attributes': self.config['features']['use_numeric_attributes'],
                'use_temporal_features': self.config['features']['use_temporal_features'],
                'use_current_seq': self.config['features']['use_current_seq'],
                'use_similar': self.config['features']['use_similar']
            },
            'model': self.config['model']
        }
    
    def create_dataset(self, df: pd.DataFrame, task_type='dynamic') -> DataLoader:
        """创建数据集和数据加载器"""
        try:
            logger.info(f"=== 开始创建{task_type}数据集 ===")
            
            if task_type == 'static':
                # 创建静态预测数据集
                dataset = self._create_static_dataset(df)
            else:
                # 创建动态预测数据集
                dataset = self._create_dynamic_dataset(df)
            
            # 创建数据加载器
            batch_size = self.config['training']['batch_size']
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            logger.info(f"{task_type}数据集创建成功，样本数: {len(dataset)}，批次大小: {batch_size}")
            return dataloader
            
        except Exception as e:
            logger.error(f"数据集创建失败: {str(e)}")
            raise
    
    def predict(self, dataloader: DataLoader, task_type='dynamic') -> np.ndarray:
        """执行预测"""
        if task_type == 'static':
            if self.static_model is None:
                raise RuntimeError("静态模型未加载，请先调用load_models('static')")
            return self._predict_static(dataloader)
        else:
            if self.dynamic_model is None:
                raise RuntimeError("动态模型未加载，请先调用load_models('dynamic')")
            return self._predict_dynamic(dataloader)
    
    def _predict_static(self, dataloader: DataLoader) -> np.ndarray:
        """执行静态预测"""
        logger.info("=== 开始静态预测 ===")
        predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                logger.info(f"=== 处理批次 {batch_idx + 1} ===")
                
                # 解包批次数据
                input_sequence, target_sequence, target, numeric_features, label_features, temporal_features, activity_text, similar_sequences = batch
                
                logger.info(f"批次数据解包完成:")
                logger.info(f"  - numeric_features: {numeric_features.shape}")
                logger.info(f"  - label_features: {label_features.shape}")
                logger.info(f"  - temporal_features: {temporal_features.shape}")
                
                # 数据转移到设备
                numeric_features = numeric_features.to(self.device)
                label_features = label_features.to(self.device)
                temporal_features = temporal_features.to(self.device)
                
                logger.info(f"数据已转移到设备: {self.device}")
                
                # 静态预测
                output = self.static_model(
                    input_sequence,
                    numeric_features,
                    label_features,
                    temporal_features,
                    activity_text,
                    similar_sequences
                )
                
                logger.info(f"静态预测输出形状: {output.shape}")
                logger.info(f"静态预测输出值范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
                logger.info(f"静态预测输出值: {output.cpu().numpy()}")
                
                predictions.append(output.cpu().numpy())
        
        # 合并预测结果
        predictions = np.concatenate(predictions, axis=0)
        
        logger.info("=== 静态预测总结 ===")
        logger.info(f"预测完成，结果形状: {predictions.shape}")
        logger.info(f"预测结果值范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
        logger.info(f"预测结果均值: {predictions.mean():.6f}")
        logger.info(f"预测结果标准差: {predictions.std():.6f}")
        logger.info("=== 静态预测完成 ===")
        
        return predictions
    
    def _predict_dynamic(self, dataloader: DataLoader) -> np.ndarray:
        """执行动态预测"""
        logger.info("=== 开始动态预测 ===")
        predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                logger.info(f"=== 处理批次 {batch_idx + 1} ===")
                
                # 解包批次数据
                input_sequence, target_sequence, target, numeric_features, label_features, temporal_features, activity_text, similar_sequences = batch
                
                logger.info(f"批次数据解包完成:")
                logger.info(f"  - input_sequence: {input_sequence.shape}")
                logger.info(f"  - numeric_features: {numeric_features.shape}")
                logger.info(f"  - label_features: {label_features.shape}")
                logger.info(f"  - temporal_features: {temporal_features.shape}")
                
                # 详细检查原始输入序列的数值
                logger.info("=== 原始输入序列详细信息 ===")
                logger.info(f"输入序列形状: {input_sequence.shape}")
                logger.info(f"输入序列数据类型: {input_sequence.dtype}")
                logger.info(f"输入序列值范围: [{input_sequence.min().item():.6f}, {input_sequence.max().item():.6f}]")
                logger.info(f"输入序列均值: {input_sequence.mean().item():.6f}")
                logger.info(f"输入序列标准差: {input_sequence.std().item():.6f}")
                
                # 输出每个时间步的详细数值
                if len(input_sequence.shape) == 2:  # [batch_size, seq_len]
                    for i in range(input_sequence.shape[0]):  # 每个样本
                        sample_values = input_sequence[i].cpu().numpy()
                        logger.info(f"样本 {i} 的14天序列值:")
                        for j, val in enumerate(sample_values):
                            logger.info(f"  第{j}天: {val:.6f}")
                elif len(input_sequence.shape) == 1:  # [seq_len]
                    sample_values = input_sequence.cpu().numpy()
                    logger.info(f"单个样本的14天序列值:")
                    for j, val in enumerate(sample_values):
                        logger.info(f"  第{j}天: {val:.6f}")
                
                logger.info("=== 原始输入序列检查完成 ===")
                
                # 数据转移到设备
                input_sequence = input_sequence.to(self.device)
                numeric_features = numeric_features.to(self.device)
                label_features = label_features.to(self.device)
                temporal_features = temporal_features.to(self.device)
                if similar_sequences is not None:
                    similar_sequences = similar_sequences.to(self.device)
                
                logger.info(f"数据已转移到设备: {self.device}")
                
                # 动态预测处理（与dynamic_exp.py保持一致）
                # 扩展输入序列维度
                input_sequence = input_sequence.unsqueeze(-1)  # [B, seq_len] -> [B, seq_len, 1]
                
                logger.info(f"输入序列扩展后形状: {input_sequence.shape}")
                logger.info(f"输入序列值范围: [{input_sequence.min().item():.6f}, {input_sequence.max().item():.6f}]")
                
                # DDN归一化处理
                if self.config['ddn']['use_ddn_normalization'] and self.ddn_model is not None:
                    logger.info("=== DDN归一化处理开始 ===")
                    
                    # 归一化
                    normalize_result = self.ddn_model.normalize(input_sequence)
                    normalized_sequence = normalize_result[0]
                    statistics_pred = normalize_result[1]
                    
                    logger.info(f"DDN归一化后序列形状: {normalized_sequence.shape}")
                    logger.info(f"DDN归一化后序列值范围: [{normalized_sequence.min().item():.6f}, {normalized_sequence.max().item():.6f}]")
                    logger.info(f"DDN统计信息: {statistics_pred}")
                    
                    # 模型预测
                    output = self.dynamic_model(
                        normalized_sequence.squeeze(-1),  # [B, seq_len, 1] -> [B, seq_len]
                        numeric_features,
                        label_features,
                        temporal_features,
                        activity_text,
                        similar_sequences
                    )
                    
                    logger.info(f"WXM模型输出形状: {output.shape}")
                    logger.info(f"WXM模型输出值范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
                    logger.info(f"WXM模型输出前5个值: {output[0, :5].cpu().numpy()}")
                    
                    # DDN反归一化
                    output = self.ddn_model.de_normalize(output.unsqueeze(-1), statistics_pred)
                    
                    logger.info(f"DDN反归一化后输出形状: {output.shape}")
                    logger.info(f"DDN反归一化后输出值范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
                    logger.info(f"DDN反归一化后输出前5个值: {output[0, :5].cpu().numpy()}")
                    
                    # 最终输出
                    output = output.squeeze(-1)  # [B, seq_len, 1] -> [B, seq_len]
                    logger.info("=== DDN处理完成 ===")
                    
                else:
                    logger.info("未使用DDN归一化，直接进行模型预测")
                    
                    output = self.dynamic_model(
                        input_sequence.squeeze(-1),
                        numeric_features,
                        label_features,
                        temporal_features,
                        activity_text,
                        similar_sequences
                    )
                    
                    logger.info(f"WXM模型输出形状: {output.shape}")
                    logger.info(f"WXM模型输出值范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
                
                logger.info(f"最终输出形状: {output.shape}")
                logger.info(f"最终输出值范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
                
                predictions.append(output.cpu().numpy())
        
        # 合并预测结果
        predictions = np.concatenate(predictions, axis=0)
        
        logger.info("=== 动态预测总结 ===")
        logger.info(f"预测完成，结果形状: {predictions.shape}")
        logger.info(f"预测结果值范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
        logger.info(f"预测结果均值: {predictions.mean():.6f}")
        logger.info(f"预测结果标准差: {predictions.std():.6f}")
        logger.info("=== 动态预测完成 ===")
        
        return predictions


# 全局预测器实例
predictor = ModelPredictor()


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """预测API端点，支持静态和动态预测"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 检查文件类型
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': '只支持CSV文件'}), 400
        
        # 获取预测类型参数
        task_type = request.form.get('task_type', 'dynamic')
        
        # 保存上传的文件到临时目录
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        try:
            # 预处理数据
            processed_df = predictor.preprocess_data(tmp_file_path, task_type)
            
            # 创建数据集
            dataloader = predictor.create_dataset(processed_df, task_type)
            
            # 执行预测
            predictions = predictor.predict(dataloader, task_type)
            
            # 格式化预测结果
            if task_type == 'static':
                result = {
                    'task_type': 'static',
                    'input_samples': len(processed_df),
                    'predictions': predictions.tolist(),
                    'config': {
                        'model_type': predictor.config['model']['model_type'],
                        'static_output_type': predictor.config.get('static_training', {}).get('static_output_type', 'value')
                    },
                    'input_columns': list(processed_df.columns),
                    'prediction_shape': list(predictions.shape),
                    'dimension_info': {
                        'description': "静态预测：基于活动属性特征预测单个数值"
                    }
                }
            else:
                result = {
                    'task_type': 'dynamic',
                    'input_samples': len(processed_df),
                    'predictions': predictions.tolist(),
                    'config': {
                        'current_seq_len': predictor.config['sequence']['current_seq_len'],
                        'predict_seq_len': predictor.config['sequence']['predict_seq_len'],
                        'model_type': predictor.config['model']['model_type'],
                        'use_ddn_normalization': predictor.config['ddn']['use_ddn_normalization']
                    },
                    'input_columns': list(processed_df.columns),
                    'prediction_shape': list(predictions.shape),
                    'dimension_info': {
                        'input_sequence_length': predictor.config['sequence']['current_seq_len'],
                        'output_sequence_length': predictor.config['sequence']['predict_seq_len'],
                        'description': f"动态预测：输入{predictor.config['sequence']['current_seq_len']}天，输出{predictor.config['sequence']['predict_seq_len']}天"
                    }
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
def load_models_api():
    """加载模型API端点"""
    try:
        data = request.get_json()
        task_type = data.get('task_type', 'dynamic') if data else 'dynamic'
        
        predictor.load_models(task_type)
        
        if task_type == 'static':
            return jsonify({
                'message': '静态预测模型加载成功',
                'device': str(predictor.device),
                'static_model_loaded': predictor.static_model is not None,
                'task_type': task_type
            })
        else:
            return jsonify({
                'message': '动态预测模型加载成功',
                'device': str(predictor.device),
                'dynamic_model_loaded': predictor.dynamic_model is not None,
                'ddn_model_loaded': predictor.ddn_model is not None,
                'use_ddn_normalization': predictor.config['ddn']['use_ddn_normalization'],
                'task_type': task_type
            })
    
    except Exception as e:
        logger.error(f"模型加载错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config_api():
    """获取当前配置API端点"""
    try:
        return jsonify(predictor.config)
    
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
        'use_ddn_normalization': predictor.config['ddn']['use_ddn_normalization']
    })


@app.route('/api/columns', methods=['GET'])
def get_required_columns():
    """获取所需的CSV列名API端点"""
    task_type = request.args.get('task_type', 'dynamic')
    
    if task_type == 'static':
        columns_info = {
            'text_features': ['activity_name', 'activity_title', 'product_names'],
            'numeric_features': ['activity_budget', 'max_reward_count', 'min_reward_count', 'duration'],
            'label_features': ['customer_id', 'template_id', 'activity_type', 'activity_form', 'bank_name', 'location', 'main_reward_type', 'secondary_reward_type', 'threshold'],
            'temporal_features': ['day', 'week', 'month', 'year'],
            'dimension_info': {
                'description': "静态预测模式：基于活动属性特征预测单个数值"
            }
        }
    else:
        columns_info = {
            'sequence_features': [str(i) for i in range(predictor.config['sequence']['current_seq_len'])],
            'text_features': ['activity_name', 'activity_title', 'product_names'],
            'numeric_features': ['activity_budget', 'max_reward_count', 'min_reward_count', 'duration'],
            'label_features': ['customer_id', 'template_id', 'activity_type', 'activity_form', 'bank_name', 'location', 'main_reward_type', 'secondary_reward_type', 'threshold'],
            'temporal_features': ['day', 'week', 'month', 'year'],
            'dimension_info': {
                'input_sequence_length': predictor.config['sequence']['current_seq_len'],
                'output_sequence_length': predictor.config['sequence']['predict_seq_len'],
                'description': f"动态预测模式：输入{predictor.config['sequence']['current_seq_len']}天历史数据，输出{predictor.config['sequence']['predict_seq_len']}天预测数据"
            }
        }
    
    return jsonify(columns_info)


@app.route('/', methods=['GET'])
def index():
    """主页"""
    return """
    <h1>WXM模型预测API</h1>
    <h2>可用端点:</h2>
    <ul>
        <li><strong>POST /api/predict</strong> - 上传CSV文件进行预测（支持静态和动态）</li>
        <li><strong>POST /api/models/load</strong> - 加载指定类型的预测模型</li>
        <li><strong>GET /api/config</strong> - 获取当前配置</li>
        <li><strong>GET /api/health</strong> - 健康检查</li>
        <li><strong>GET /api/columns</strong> - 获取所需的CSV列名</li>
    </ul>
    
    <h2>预测模式说明:</h2>
    
    <h3>静态预测模式:</h3>
    <ul>
        <li>输入: 活动属性特征（文本、数值、标签、时间特征）</li>
        <li>输出: 单个数值预测结果</li>
        <li>适用场景: 基于活动属性预测活动效果指标</li>
    </ul>
    
    <h3>动态预测模式:</h3>
    <ul>
        <li>输入: {current_seq_len}天的历史序列数据 + 活动属性特征</li>
        <li>输出: {predict_seq_len}天的预测序列数据</li>
        <li>CSV格式要求: 前{current_seq_len}列(0-{current_seq_len-1})必须为数值型序列数据，列名必须为"0","1","2",...,"{current_seq_len-1}"</li>
        <li>其余列为活动属性特征（文本、数值、标签、时间特征）</li>
        <li>适用场景: 基于历史序列和活动属性预测未来趋势</li>
    </ul>
    
    <h2>使用示例:</h2>
    <pre>
    # 加载静态预测模型
    curl -X POST http://localhost:5000/api/models/load \
         -H "Content-Type: application/json" \
         -d '{"task_type": "static"}'
    
    # 加载动态预测模型
    curl -X POST http://localhost:5000/api/models/load \
         -H "Content-Type: application/json" \
         -d '{"task_type": "dynamic"}'
    
    # 进行静态预测
    curl -X POST http://localhost:5000/api/predict \
         -F "file=@your_data.csv" \
         -F "task_type=static"
    
    # 进行动态预测
    curl -X POST http://localhost:5000/api/predict \
         -F "file=@your_data.csv" \
         -F "task_type=dynamic"
    
    # 获取静态预测所需列名
    curl "http://localhost:5000/api/columns?task_type=static"
    
    # 获取动态预测所需列名
    curl "http://localhost:5000/api/columns?task_type=dynamic"
    </pre>
    """.format(
        current_seq_len=predictor.config['sequence']['current_seq_len'],
        predict_seq_len=predictor.config['sequence']['predict_seq_len']
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 