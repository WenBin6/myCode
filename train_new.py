#!/usr/bin/env python3
"""
新的训练脚本
使用配置文件管理参数，整合优化后的模块
"""
import os
import yaml
import argparse
import pandas as pd
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.data_processor import DataProcessor, UnifiedDataset
from utils.static_exp import StaticExp
from utils.dynamic_exp import DynamicExp
from models.WXM import WXM
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CURL_CA_BUNDLE'] = ''


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: str):
    """保存配置到JSON格式（用于兼容现有代码）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def convert_config_to_args(config: dict) -> argparse.Namespace:
    """将配置字典转换为argparse.Namespace对象（用于兼容现有代码）"""
    args = argparse.Namespace()
    
    # 基础配置
    args.random_seed = config.get('random_seed', 2024)
    args.gpu_num = config.get('gpu_num', 0)
    args.log_dir = config.get('log_dir', 'log')
    
    # 数据路径配置
    data_config = config.get('data', {})
    args.oringinal_data_path = data_config.get('original_data_path', 'data/activity_order_independent_id')
    args.historical_activities_path = data_config.get('historical_activities_path', 'data')
    args.activities_attributes_path = data_config.get('activities_attributes_path', 'data')
    args.dataset_df_path = data_config.get('dataset_df_path', 'data/dataset_df.csv')
    
    # 预测目标配置
    prediction_config = config.get('prediction', {})
    args.freq = prediction_config.get('freq', 'd')
    args.target = prediction_config.get('target', 'uc')
    args.static_target = prediction_config.get('static_target', 'user_sum')
    
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

    # 静态预测训练参数
    static_training_config = config.get('static_training', {})
    args.static_epochs = static_training_config.get('static_epochs', 10)
    args.static_learning_rate = static_training_config.get('static_learning_rate', 0.1)
    args.static_output_type = static_training_config.get('static_output_type', 'quantile')
    args.num_gated_blocks = static_training_config.get('num_gated_blocks', 2)

    # 动态预测训练参数
    dynamic_training_config = config.get('dynamic_training', {})
    args.dynamic_epochs = dynamic_training_config.get('dynamic_epochs', 100)
    args.dynamic_learning_rate = dynamic_training_config.get('dynamic_learning_rate', 0.1)
    
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
    
    # 其他参数
    args.process_data = False  # 默认不重新处理数据
    args.task_type = 'static'  # 默认静态预测，可通过命令行参数覆盖
    
    return args


def run_training(config_path: str, task_type: str = 'static', process_data: bool = False):
    """运行训练"""
    print(f"使用配置文件: {config_path}")
    print(f"任务类型: {task_type}")
    
    # 加载配置
    config = load_config(config_path)
    args = convert_config_to_args(config)
    args.task_type = task_type
    args.process_data = process_data
    
    print("配置参数:")
    print(f"  任务类型: {args.task_type}")
    print(f"  模型类型: {args.model_type}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.static_learning_rate if task_type == 'static' else args.dynamic_learning_rate}")
    
    # 设置随机种子
    StaticExp.set_seed(args.random_seed)
    
    # 数据处理
    data_processor = DataProcessor(config)
    
    # 检查是否需要重新处理数据
    need_reprocess = args.process_data
    
    # 检查数据集是否存在
    if os.path.exists(args.dataset_df_path):
        # 如果数据集存在，检查是否包含序列数据
        dataset_df = pd.read_csv(args.dataset_df_path)
    else:
        print(f"数据集文件不存在: {args.dataset_df_path}")
        need_reprocess = True
    
    # 如果需要重新处理数据
    if need_reprocess:
        print("开始处理原始数据...")
        data_processor.process_activities(
            args.oringinal_data_path, 
            args.historical_activities_path, 
            args.freq
        )
        
        # 处理完成后，创建数据集
        print("创建数据集...")
        dataset_df = data_processor.create_dataset(
            task_type=task_type,
            sequence_process_type=args.sequence_process_type,
            current_seq_len=args.current_seq_len,
            predict_seq_len=args.predict_seq_len,
            total_seq_len=args.total_seq_len,
            oringinal_data_path=args.oringinal_data_path,
            historical_activities_path=args.historical_activities_path,
            target=args.target,
            freq=args.freq,
            apply_diff=args.apply_diff,
            apply_smoothing=args.apply_smoothing,
            activities_attributes_path=args.activities_attributes_path
        )
        
        # 保存数据集
        os.makedirs(os.path.dirname(args.dataset_df_path), exist_ok=True)
        dataset_df.to_csv(args.dataset_df_path, index=False)
        print(f"数据集已保存到: {args.dataset_df_path}")
    
    # 再次检查数据集是否存在
    if not os.path.exists(args.dataset_df_path):
        print(f"数据集文件不存在: {args.dataset_df_path}")
        print("数据处理失败，请检查原始数据路径")
        return
    
    # 加载数据集
    print("加载数据集...")
    dataset_df = pd.read_csv(args.dataset_df_path)
    print(f"数据集大小: {len(dataset_df)} 行")
    
    # 打印数据集详细信息
    print("\n" + "="*50)
    print("数据集详细信息")
    print("="*50)
    print(f"数据集文件: {args.dataset_df_path}")
    print(f"总样本数: {len(dataset_df)}")
    print(f"特征数量: {len(dataset_df.columns)}")
    print(f"任务类型: {task_type}")
    
    # 打印目标变量信息
    if task_type == 'static':
        target_col = args.static_target
        print(f"目标变量: {target_col}")
        if target_col in dataset_df.columns:
            target_stats = dataset_df[target_col].describe()
            print(f"目标变量统计:")
            print(f"  均值: {target_stats['mean']:.2f}")
            print(f"  标准差: {target_stats['std']:.2f}")
            print(f"  最小值: {target_stats['min']:.2f}")
            print(f"  最大值: {target_stats['max']:.2f}")
    else:
        print(f"目标变量: {args.target} (时间序列)")
        print(f"序列长度: {args.current_seq_len} + {args.predict_seq_len} = {args.current_seq_len + args.predict_seq_len}")
        
        # 检查序列数据列
        sequence_columns = [col for col in dataset_df.columns if 'seq_' in col or 'sequence' in col]
        if sequence_columns:
            print(f"序列数据列: {len(sequence_columns)} 个")
            for i, col in enumerate(sequence_columns[:5]):  # 只显示前5个
                print(f"  {i+1}. {col}")
            if len(sequence_columns) > 5:
                print(f"  ... 还有 {len(sequence_columns) - 5} 个序列列")
    
    # 打印特征信息
    print(f"\n特征列:")
    for i, col in enumerate(dataset_df.columns[:10]):  # 只显示前10列
        print(f"  {i+1:2d}. {col}")
    if len(dataset_df.columns) > 10:
        print(f"  ... 还有 {len(dataset_df.columns) - 10} 个特征")
    
    # 检查缺失值
    missing_values = dataset_df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n缺失值情况:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"  {col}: {missing} 个缺失值")
    else:
        print(f"\n缺失值: 无")
    
    print("="*50 + "\n")
    
    # 分割数据集
    train_dataset_df, test_dataset_df = train_test_split(
        dataset_df, test_size=0.2, shuffle=False, random_state=args.random_seed
    )
    
    # 创建数据加载器
    sequence_len = args.current_seq_len + args.predict_seq_len if task_type == 'dynamic' else args.total_seq_len
    
    train_dataset = UnifiedDataset(
        train_dataset_df, config, task_type, 'train'
    )
    test_dataset = UnifiedDataset(
        test_dataset_df, config, task_type, 'test'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    print("创建模型...")
    model = WXM(args)
    
    # 选择实验类型
    if task_type == 'static':
        exp = StaticExp(args)
        epochs = args.static_epochs
    else:
        exp = DynamicExp(args)
        epochs = args.dynamic_epochs
    
    # 运行训练
    print(f"开始训练，共 {epochs} 个epoch...")
    predictions, true_values = exp.run(train_loader, test_loader)
    
    print("训练完成！")
    return predictions, true_values


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态活动预测系统')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--task_type', type=str, default='dynamic', 
                       choices=['static', 'dynamic'], help='任务类型')
    parser.add_argument('--process_data', action='store_true', 
                       help='是否重新处理数据，通常修改task_type、current_seq_len、predict_seq_len、freq时需要重新处理数据')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        print("请确保配置文件存在或使用 --config 参数指定正确的路径")
        return
    
    # 检查数据集是否存在，如果不存在且是动态预测，提示用户
    config = load_config(args.config)
    args_config = convert_config_to_args(config)
    dataset_path = args_config.dataset_df_path
    
    if not os.path.exists(dataset_path) and args.task_type == 'dynamic':
        print(f"注意: 数据集文件不存在 ({dataset_path})")
        print("动态预测需要序列数据，将自动重新处理数据...")
        args.process_data = True
    
    # 运行训练
    run_training(args.config, args.task_type, args.process_data)


if __name__ == '__main__':
    main() 