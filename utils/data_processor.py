"""
统一的数据处理模块
整合所有数据处理功能，避免代码重复
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pad_sequence
import pickle
from typing import Dict, List, Tuple, Optional


class DataProcessor:
    """统一的数据处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.label_encoders = {}
        
    def process_activities(self, original_data_path: str, historical_activities_path: str, freq: str) -> None:
        """处理活动序列数据，生成活动统计数据"""
        print(f"开始处理活动数据，频率: {freq}")
        
        activities_duration = []
        processed_files_count = 0
        total_files = len([f for f in os.listdir(original_data_path) if f.endswith('.csv')])
        print(f"总共发现 {total_files} 个CSV文件")
        
        for i, file_name in enumerate(os.listdir(original_data_path)):
            if not file_name.endswith('.csv'):
                continue
                
            activity_id = file_name[:-4]
            file_path = os.path.join(original_data_path, file_name)
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            print(f"[{i+1}/{total_files}] 处理文件: {file_name} (大小: {file_size/1024/1024:.2f}MB)")
            
            # 跳过过大的文件（超过100MB）
            if file_size > 100 * 1024 * 1024:  # 100MB
                print(f"文件过大，跳过: {file_name}")
                continue
            
            # 尝试多种方式读取CSV文件
            data = None
            try:
                # 首先尝试标准方式读取
                data = pd.read_csv(file_path)
                print(f"  成功读取文件: {file_name}")
            except Exception as e1:
                print(f"  标准读取失败，尝试使用python引擎: {file_name}")
                try:
                    # 尝试使用python引擎
                    data = pd.read_csv(file_path, engine='python')
                    print(f"  Python引擎读取成功: {file_name}")
                except Exception as e2:
                    print(f"  Python引擎读取失败，尝试指定编码: {file_name}")
                    try:
                        # 尝试指定编码
                        data = pd.read_csv(file_path, engine='python', encoding='utf-8')
                        print(f"  UTF-8编码读取成功: {file_name}")
                    except Exception as e3:
                        print(f"  UTF-8编码读取失败，尝试GBK编码: {file_name}")
                        try:
                            # 尝试GBK编码
                            data = pd.read_csv(file_path, engine='python', encoding='gbk')
                            print(f"  GBK编码读取成功: {file_name}")
                        except Exception as e4:
                            print(f"  所有读取方式都失败，跳过文件: {file_name}")
                            print(f"  错误信息: {e4}")
                            continue
            
            if data is None or data.empty:
                print(f"  文件 {file_name} 数据为空，跳过处理")
                continue
                
            # 检查必要的列是否存在
            required_columns = ['create_date', 'activity_name']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"  文件 {file_name} 缺少必要列: {missing_columns}，跳过处理")
                continue
            
            try:
                data['create_date'] = pd.to_datetime(data['create_date'])
                print(f"  日期转换成功: {file_name}")
            except Exception as e:
                print(f"  转换日期失败: {file_name}, 错误: {e}")
                continue
            
            if data['activity_name'].isnull().all():
                print(f"  文件 {file_name} 活动名称为空，跳过处理")
                continue
                
            activity_name = data['activity_name'].iloc[1] if len(data) > 1 else data['activity_name'].iloc[0]
            
            # 按频率聚合数据
            try:
                processed_data = self._aggregate_by_freq(data, freq)
                print(f"  数据聚合成功: {file_name}")
            except Exception as e:
                print(f"  聚合数据失败: {file_name}, 错误: {e}")
                continue
            
            # 保存处理后的数据
            processed_data_path = f"{original_data_path}_{freq}"
            os.makedirs(processed_data_path, exist_ok=True)
            output_file_path = os.path.join(processed_data_path, f"{activity_id}_{freq}.csv")
            processed_data.to_csv(output_file_path, index=False)
            
            # 计算统计信息
            oc_sum = processed_data['oc'].sum()
            user_sum = processed_data['uc'].sum()
            duration = (processed_data['oc'] > 0).sum()
            
            activities_duration.append({
                'activity_id': activity_id,
                'activity_name': activity_name,
                'duration': duration,
                'oc_sum': oc_sum,
                'user_sum': user_sum
            })
            processed_files_count += 1
            print(f"  处理完成: {file_name}")
            
        # 保存活动持续时间数据
        if activities_duration:
            activities_duration_df = pd.DataFrame(activities_duration)
            activities_duration_df.to_csv(
                os.path.join(historical_activities_path, f'activities_duration_{freq}.csv'), 
                index=False
            )
            print(f"成功处理 {processed_files_count} 个文件")
        else:
            print("没有成功处理任何文件")
    
    def _aggregate_by_freq(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """按频率聚合数据"""
        data.set_index('create_date', inplace=True)
        
        freq_mapping = {
            'h': ('H', 'H'),
            'd': ('D', 'D'), 
            'w': ('W', 'W'),
            'm': ('ME', 'ME')
        }
        
        if freq not in freq_mapping:
            raise ValueError(f"不支持的频率: {freq}")
            
        resample_freq, range_freq = freq_mapping[freq]
        
        processed_data = data.resample(resample_freq).agg({
            'id': 'size', 
            'user_id': pd.Series.nunique
        }).reset_index()
        
        # 补齐缺失日期
        full_range = pd.date_range(
            start=processed_data['create_date'].min(),
            end=processed_data['create_date'].max(), 
            freq=range_freq
        )
        processed_data = processed_data.set_index('create_date').reindex(
            full_range, fill_value=0
        ).rename_axis('create_date').reset_index()
        
        processed_data.rename(columns={'id': 'oc', 'user_id': 'uc'}, inplace=True)
        return processed_data
    
    def create_dataset(self, task_type: str, **kwargs) -> pd.DataFrame:
        """创建数据集"""
        if task_type == 'static':
            return self._create_static_dataset(**kwargs)
        elif task_type == 'dynamic':
            return self._create_dynamic_dataset(**kwargs)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
    
    def _create_static_dataset(self, **kwargs) -> pd.DataFrame:
        """创建静态预测数据集（迁移自 enrich_dataset_with_attributes 相关逻辑）"""
        # 1. 先用 batch_process_sequences 生成活动ID列表
        task_type = 'static'
        sequence_process_type = kwargs.get('sequence_process_type', 'LTTB')
        current_days = kwargs.get('current_seq_len', 14)
        predict_days = kwargs.get('predict_seq_len', 7)
        total_days = kwargs.get('total_seq_len', 30)
        oringinal_data_path = kwargs.get('oringinal_data_path', self.config['data']['original_data_path'])
        historical_activities_path = kwargs.get('historical_activities_path', self.config['data']['historical_activities_path'])
        target = kwargs.get('target', self.config['prediction']['target'])
        freq = kwargs.get('freq', self.config['prediction']['freq'])
        apply_diff = kwargs.get('apply_diff', self.config['sequence']['apply_diff'])
        apply_smoothing = kwargs.get('apply_smoothing', self.config['sequence']['apply_smoothing'])
        activities_attributes_path = kwargs.get('activities_attributes_path', self.config['data']['activities_attributes_path'])

        # 只生成活动ID
        processed_data_path_with_freq = f"{oringinal_data_path}_{freq}"
        result_dataset = pd.DataFrame()
        historical_activities = pd.read_csv(historical_activities_path + f'/activities_duration_{freq}.csv')
        for activity_id in historical_activities['activity_id'].unique():
            encoded_df = pd.DataFrame({'activity_id': [activity_id]})
            result_dataset = pd.concat([result_dataset, encoded_df], ignore_index=True)
        # enrich_dataset_with_attributes
        activity_duration = pd.read_csv(historical_activities_path + f'/activities_duration_{freq}.csv')
        attributes_file = os.path.join(activities_attributes_path, 'all_activities_attributes_threshold.csv')
        activity_attributes = pd.read_csv(attributes_file)
        # 去除重复列
        for col in ['oc_sum', 'user_sum', 'duration']:
            if col in activity_attributes.columns:
                activity_attributes = activity_attributes.drop(columns=[col])
        activity_attributes = pd.merge(activity_attributes, activity_duration[['activity_id', 'oc_sum', 'user_sum','duration']], left_on='id', right_on='activity_id', how='left').drop(columns=['activity_id'])
        enriched_dataset = pd.merge(result_dataset, activity_attributes, left_on='activity_id', right_on='id', how='left').drop(columns=['id'])
        # 排序（如果有 activity_start_time）
        if 'activity_start_time' in enriched_dataset.columns:
            enriched_dataset = enriched_dataset.sort_values(by='activity_start_time')
        return enriched_dataset

    def _create_dynamic_dataset(self, **kwargs) -> pd.DataFrame:
        """创建动态预测数据集（迁移自 batch_process_sequences + enrich_dataset_with_attributes 逻辑）"""
        task_type = 'dynamic'
        sequence_process_type = kwargs.get('sequence_process_type', 'LTTB')
        current_days = kwargs.get('current_seq_len', 14)
        predict_days = kwargs.get('predict_seq_len', 7)
        total_days = kwargs.get('total_seq_len', 30)
        oringinal_data_path = kwargs.get('oringinal_data_path', self.config['data']['original_data_path'])
        historical_activities_path = kwargs.get('historical_activities_path', self.config['data']['historical_activities_path'])
        target = kwargs.get('target', self.config['prediction']['target'])
        freq = kwargs.get('freq', self.config['prediction']['freq'])
        apply_diff = kwargs.get('apply_diff', self.config['sequence']['apply_diff'])
        apply_smoothing = kwargs.get('apply_smoothing', self.config['sequence']['apply_smoothing'])
        activities_attributes_path = kwargs.get('activities_attributes_path', self.config['data']['activities_attributes_path'])

        processed_data_path_with_freq = f"{oringinal_data_path}_{freq}"
        result_dataset = pd.DataFrame()
        historical_activities = pd.read_csv(historical_activities_path + f'/activities_duration_{freq}.csv')
        for activity_id in historical_activities['activity_id'].unique():
            file_path = os.path.join(processed_data_path_with_freq, f"{activity_id}_{freq}.csv")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                activity_data = pd.read_csv(file_path)
                activity_length = activity_data.shape[0]
                if activity_length < current_days + predict_days:
                    continue
                sequence_array = activity_data[target].to_numpy().astype(float)
                encoded_df = pd.DataFrame(sequence_array[:current_days + predict_days].reshape(1, -1))
                encoded_df['activity_id'] = activity_id
                result_dataset = pd.concat([result_dataset, encoded_df], ignore_index=True)
            else:
                continue
        # enrich_dataset_with_attributes
        activity_duration = pd.read_csv(historical_activities_path + f'/activities_duration_{freq}.csv')
        attributes_file = os.path.join(activities_attributes_path, 'all_activities_attributes_threshold.csv')
        activity_attributes = pd.read_csv(attributes_file)
        for col in ['oc_sum', 'user_sum', 'duration']:
            if col in activity_attributes.columns:
                activity_attributes = activity_attributes.drop(columns=[col])
        activity_attributes = pd.merge(activity_attributes, activity_duration[['activity_id', 'oc_sum', 'user_sum','duration']], left_on='id', right_on='activity_id', how='left').drop(columns=['activity_id'])
        enriched_dataset = pd.merge(result_dataset, activity_attributes, left_on='activity_id', right_on='id', how='left').drop(columns=['id'])
        if 'activity_start_time' in enriched_dataset.columns:
            enriched_dataset = enriched_dataset.sort_values(by='activity_start_time')
        return enriched_dataset


class UnifiedDataset(Dataset):
    """统一的数据集类，支持静态和动态预测"""
    
    def __init__(self, data_df: pd.DataFrame, config: Dict, task_type: str = 'static', 
                 mode: str = 'train'):
        self.data_df = data_df.copy()
        self.config = config
        self.task_type = task_type
        self.mode = mode
        self.label_encoders = {}  # 添加缺失的属性
        
        # 根据任务类型设置序列长度
        if task_type == 'static':
            self.sequence_len = config['sequence']['total_seq_len']
            self.current_seq_len = config['sequence']['total_seq_len']
        else:  # dynamic
            self.sequence_len = (config['sequence']['current_seq_len'] + 
                               config['sequence']['predict_seq_len'])
            self.current_seq_len = config['sequence']['current_seq_len']
        
        self.preprocess_data()
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        if self.task_type == 'dynamic':
            # 动态预测：分离输入序列和目标序列
            # 确保数据类型正确
            input_values = row.iloc[:self.current_seq_len].values
            target_values = row.iloc[self.current_seq_len:self.sequence_len].values
            
            # 转换为float32类型
            try:
                input_sequence = torch.tensor(input_values.astype(np.float32), dtype=torch.float32)
                target_sequence = torch.tensor(target_values.astype(np.float32), dtype=torch.float32)
            except (ValueError, TypeError) as e:
                print(f"数据类型转换错误: {e}")
                print(f"输入序列数据类型: {input_values.dtype}")
                print(f"目标序列数据类型: {target_values.dtype}")
                # 使用默认值
                input_sequence = torch.zeros(self.current_seq_len, dtype=torch.float32)
                target_sequence = torch.zeros(self.sequence_len - self.current_seq_len, dtype=torch.float32)
        else:
            # 静态预测：不需要序列数据
            input_sequence = torch.tensor([])
            target_sequence = torch.tensor([])
        
        # 提取特征
        numeric_features = self._extract_numeric_features(row)
        label_features = self._extract_label_features(row)
        temporal_features = self._extract_temporal_features(row)
        activity_text = self._extract_text_features(row)
        similar_sequences = self._extract_similar_sequences(row)
        
        # 目标值
        if self.task_type == 'static':
            target = torch.tensor(row[self.config['prediction']['static_target']], dtype=torch.float32)
        else:
            target = torch.tensor(0.0)  # 动态预测不需要标量目标
        
        return (input_sequence, target_sequence, target, numeric_features, 
                label_features, temporal_features, activity_text, similar_sequences)
    
    def _extract_numeric_features(self, row) -> torch.Tensor:
        """提取数值特征"""
        return torch.tensor([
            row['activity_budget'],
            row['max_reward_count'], 
            row['min_reward_count'],
            row['duration']
        ], dtype=torch.float32)
    
    def _extract_label_features(self, row) -> torch.Tensor:
        """提取标签特征"""
        return torch.tensor([
            row['customer_id_encoded'],
            row['activity_type_encoded'],
            row['activity_form_encoded'],
            row['bank_name_encoded'],
            row['location_encoded'],
            row['main_reward_type_encoded'],
            row['secondary_reward_type_encoded'],
            row['template_id_encoded'],
            row.get('threshold', 0)  # 兼容性处理
        ], dtype=torch.float32)
    
    def _extract_temporal_features(self, row) -> torch.Tensor:
        """提取时间特征"""
        return torch.tensor(
            row[['day', 'week', 'month', 'year']].astype(float).values,
            dtype=torch.float32
        )
    
    def _extract_text_features(self, row) -> Dict[str, str]:
        """提取文本特征"""
        return {
            'activity_name': row['activity_name'],
            'activity_title': row['activity_title'],
            'product_names': row['product_names']
        }
    
    def _extract_similar_sequences(self, row) -> torch.Tensor:
        """提取相似序列"""
        similar_sequences = []
        for i in range(self.sequence_len):
            similar_column = f'similar_{i}'
            if similar_column in row:
                similar_sequences.append(row[similar_column])
            else:
                similar_sequences.append(np.nan)
        return torch.tensor(similar_sequences, dtype=torch.float32)
    
    def preprocess_data(self):
        """数据预处理"""
        # 异常值处理
        if self.task_type == 'static':
            self._remove_outliers()
        
        # 特征编码
        self._encode_categorical_features()
        
        # 序列处理
        if self.task_type == 'dynamic' and self.config['sequence']['series_scale']:
            self._add_similar_sequences()
            self._scale_sequences()
        
        # 特征归一化
        if self.config['sequence']['feature_scale']:
            self._scale_features()
    
    def _remove_outliers(self):
        """去除异常值"""
        columns_of_interest = [
            'max_reward_count', 'min_reward_count',
            'activity_budget', 'duration', 'oc_sum', 'user_sum'
        ]
        self.data_df = self._remove_outliers_zscore(self.data_df, columns_of_interest)
    
    def _encode_categorical_features(self):
        """编码分类特征"""
        categorical_columns = [
            'customer_id', 'template_id', 'activity_type', 'activity_form',
            'bank_name', 'location', 'main_reward_type', 'secondary_reward_type'
        ]
        
        for col in categorical_columns:
            if col in self.data_df.columns:
                le = LabelEncoder()
                self.data_df[f'{col}_encoded'] = le.fit_transform(self.data_df[col])
                self.label_encoders[col] = le
    
    def _add_similar_sequences(self):
        """添加相似序列"""
        self.data_df = self._find_similar_sequences(self.data_df, self.sequence_len)
    
    def _scale_sequences(self):
        """序列归一化"""
        time_series_data = self.data_df.iloc[:, :self.sequence_len]
        for i in range(len(self.data_df)):
            sequence_scaler = StandardScaler()
            scaled_sequence = sequence_scaler.fit_transform(
                time_series_data.iloc[i, :self.current_seq_len].values.reshape(-1, 1)
            ).flatten()
            self.data_df.iloc[i, :self.current_seq_len] = scaled_sequence
    
    def _scale_features(self):
        """特征归一化"""
        scaler = StandardScaler()
        numeric_columns = [
            'activity_budget', 'max_reward_count', 'min_reward_count', 'duration'
        ]
        self.data_df[numeric_columns] = scaler.fit_transform(self.data_df[numeric_columns])
    
    @staticmethod
    def _remove_outliers_zscore(data: pd.DataFrame, columns: List[str], threshold: float = 3) -> pd.DataFrame:
        """使用Z-score去除异常值"""
        data_clean = data.copy()
        for col in columns:
            if col in data_clean.columns:
                z_scores = np.abs((data_clean[col] - data_clean[col].mean()) / data_clean[col].std())
                data_clean = data_clean[z_scores < threshold]
        return data_clean
    
    @staticmethod
    def _find_similar_sequences(df: pd.DataFrame, sequence_len: int) -> pd.DataFrame:
        """找到相似序列"""
        df = df.reset_index(drop=True)
        sequence_columns = df.columns[:sequence_len]
        
        similar_data = pd.DataFrame(
            index=df.index, 
            columns=[f'similar_{i}' for i in range(sequence_len)]
        )
        similar_data = similar_data.fillna(np.nan)
        
        for row_index in range(len(df)):
            similarities = cosine_similarity(
                df[sequence_columns], 
                df.loc[row_index, sequence_columns].values.reshape(1, -1)
            )
            similarities[row_index] = -1
            similar_row_index = np.argmax(similarities)
            similar_sequence = df.iloc[similar_row_index, :sequence_len].values
            similar_data.iloc[row_index, :sequence_len] = similar_sequence
        
        return pd.concat([df, similar_data], axis=1) 