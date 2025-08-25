import math
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet50, ResNet50_Weights

from layers.RevIN import RevIN


@dataclass
class ModelConfig:
    """模型配置类，统一管理所有模型参数"""
    
    # 基础配置
    embedding_dim: int = 128
    hidden_dim: int = 256
    d_model: int = 128
    d_s: int = 64
    d_n: int = 64
    
    # 注意力配置
    num_heads: int = 8
    num_layers: int = 2
    
    # 序列配置
    current_seq_len: int = 14
    predict_seq_len: int = 7
    total_seq_len: int = 30
    input_dim: int = 1
    
    # 功能开关
    use_img: bool = False
    use_text_attributes: bool = True
    use_label_attributes: bool = True
    use_numeric_attributes: bool = True
    use_temporal_features: bool = True
    use_current_seq: bool = True
    use_similar: bool = False
    use_encoder_mask: bool = False
    autoregressive: bool = False
    
    # 训练配置
    dropout: float = 0.1
    gpu_num: int = 0
    task_type: str = 'static'  # 'static' or 'dynamic'


class TextEmbedder(nn.Module):
    """文本嵌入模块，使用BERT进行文本特征提取"""
    
    def __init__(self, embedding_dim: int, gpu_num: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gpu_num = gpu_num
        
        # BERT模型初始化（使用本地模型文件）
        bert_model_path = os.path.join(os.path.dirname(__file__), '..', 'bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        
        # 投影层
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 设备配置
        self.device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, activity_name: List[str], activity_title: List[str], 
                product_names: List[str]) -> torch.Tensor:
        """前向传播
        
        Args:
            activity_name: 活动名称列表
            activity_title: 活动标题列表  
            product_names: 产品名称列表
            
        Returns:
            torch.Tensor: 文本嵌入特征 [batch_size, embedding_dim]
        """
        # 文本拼接
        textual_descriptions = [
            f"{name} {title} {product}" 
            for name, title, product in zip(activity_name, activity_title, product_names)
        ]
        
        # BERT编码
        inputs = self.tokenizer(
            textual_descriptions, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # 设备转移
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # BERT前向传播
        with torch.no_grad():  # 冻结BERT参数
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        
        # 特征提取和投影
        pooled_output = outputs.pooler_output
        embeddings = self.dropout(self.fc(pooled_output))
        
        return embeddings


class ImageEmbedder(nn.Module):
    """图像嵌入模块，使用ResNet50提取图像特征"""
    
    def __init__(self):
        super().__init__()
        # 图像特征提取
        weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = resnet50(weights=weights)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # 冻结ResNet参数
        for p in self.resnet.parameters():
            p.requires_grad = False
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            images: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 图像特征 [batch_size, 2048, h/32, w/32]
        """
        img_embeddings = self.resnet(images)
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2], -1)
        return out.view(*size).contiguous()


class DummyEmbedder(nn.Module):
    """时间特征嵌入模块，处理日、周、月、年等时间特征"""
    
    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 各时间维度嵌入层
        self.time_embeddings = nn.ModuleDict({
            'day': nn.Linear(1, embedding_dim),
            'week': nn.Linear(1, embedding_dim), 
            'month': nn.Linear(1, embedding_dim),
            'year': nn.Linear(1, embedding_dim)
        })
        
        # 特征融合层
        self.fusion_layer = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            temporal_features: 时间特征 [batch_size, 4] (day, week, month, year)
            
        Returns:
            torch.Tensor: 时间嵌入特征 [batch_size, embedding_dim]
        """
        temporal_features = temporal_features.float()
        
        # 特征分离
        day_feat = temporal_features[:, 0:1]    # [batch_size, 1]
        week_feat = temporal_features[:, 1:2]   # [batch_size, 1]
        month_feat = temporal_features[:, 2:3]  # [batch_size, 1]
        year_feat = temporal_features[:, 3:4]   # [batch_size, 1]
        
        # 各维度嵌入
        embeddings = {
            'day': self.time_embeddings['day'](day_feat),
            'week': self.time_embeddings['week'](week_feat),
            'month': self.time_embeddings['month'](month_feat),
            'year': self.time_embeddings['year'](year_feat)
        }
        
        # 特征融合
        concatenated = torch.cat(list(embeddings.values()), dim=1)
        fused_embeddings = self.dropout(self.fusion_layer(concatenated))
        
        return fused_embeddings


class CrossModalAttention(nn.Module):
    """跨模态注意力模块，用于融合不同模态的信息"""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            query: 查询张量 [N, embed_dim]
            key: 键张量 [N, embed_dim]
            value: 值张量 [N, embed_dim]
            
        Returns:
            torch.Tensor: 注意力输出 [N, embed_dim]
        """
        attn_output, _ = self.multihead_attn(
            query.unsqueeze(0),  # [1, N, D]
            key.unsqueeze(0),    # [1, N, D]
            value.unsqueeze(0)   # [1, N, D]
        )
        return self.layer_norm(query + attn_output.squeeze(0))


class CombinedModel(nn.Module):
    """多模态融合模型，整合文本、数值、标签和时间特征"""
    
    def __init__(self, use_img: bool, use_text_attributes: bool, 
                 use_label_attributes: bool, use_numeric_attributes: bool,
                 use_temporal_features: bool, embedding_dim: int, 
                 hidden_dim: int, num_heads: int, gpu_num: int, dropout: float):
        super().__init__()
        
        # 参数配置
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_img = use_img
        self.use_text_attributes = use_text_attributes
        self.use_label_attributes = use_label_attributes
        self.use_numeric_attributes = use_numeric_attributes
        self.use_temporal_features = use_temporal_features
        
        # 初始化编码器和融合模块
        self._init_encoders(embedding_dim)
        self.cross_attentions = self._build_cross_attention_modules()
        self.final_proj = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _init_encoders(self, embed_dim: int):
        """初始化各模态编码器"""
        self.label_feature_fc = nn.Sequential(
            nn.Linear(9, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.numeric_feature_fc = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.temporal_feature_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def _build_cross_attention_modules(self) -> nn.ModuleDict:
        """动态构建交叉注意力模块"""
        modules = nn.ModuleDict()
        active_modalities = []
        
        # 记录激活的模态
        if self.use_text_attributes: 
            active_modalities.append('text')
        if self.use_label_attributes: 
            active_modalities.append('label')
        if self.use_numeric_attributes: 
            active_modalities.append('numeric')
        if self.use_temporal_features: 
            active_modalities.append('temporal')
        if self.use_img: 
            active_modalities.append('image')
        
        # 构建所有两两交叉注意力组合
        for i in range(len(active_modalities)):
            for j in range(i+1, len(active_modalities)):
                mod1, mod2 = active_modalities[i], active_modalities[j]
                modules[f"{mod1}_{mod2}"] = CrossModalAttention(
                    self.embedding_dim, self.num_heads
                )
        return modules

    def _cross_modal_fusion(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行跨模态注意力融合"""
        if not embeddings:
            raise ValueError("embeddings字典不能为空")
            
        fused_feature = torch.zeros_like(next(iter(embeddings.values())))
        count = 0.0
        
        # 遍历所有预定义的交叉注意力组合
        for name in self.cross_attentions:
            mod1, mod2 = name.split('_')
            query = embeddings[mod1]
            key = embeddings[mod2]
            
            # 双向注意力计算
            fusion_1 = self.cross_attentions[name](query, key, key)
            fusion_2 = self.cross_attentions[name](key, query, query)
            
            fused_feature += fusion_1 + fusion_2
            count += 2.0
        
        # 平均融合
        if count > 0:
            return fused_feature / count
        else:
            # 确保返回tensor类型
            return torch.stack(list(embeddings.values())).sum(dim=0)

    def forward(self, numeric_features: torch.Tensor, label_features: torch.Tensor,
               text_embeddings: torch.Tensor, dummy_embeddings: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            numeric_features: 数值特征 [batch_size, 4]
            label_features: 标签特征 [batch_size, 9]
            text_embeddings: 文本嵌入 [batch_size, embedding_dim]
            dummy_embeddings: 时间嵌入 [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: 融合后的特征 [batch_size, hidden_dim]
        """
        # 编码各模态特征
        embeddings = {}
        
        if self.use_text_attributes:
            embeddings['text'] = text_embeddings
            
        if self.use_label_attributes:
            embeddings['label'] = self.label_feature_fc(label_features)
            
        if self.use_numeric_attributes:
            embeddings['numeric'] = self.numeric_feature_fc(numeric_features)
            
        if self.use_temporal_features:
            embeddings['temporal'] = self.temporal_feature_fc(dummy_embeddings)
        
        # 跨模态融合
        if len(embeddings) > 1:
            fused_feature = self._cross_modal_fusion(embeddings)
        else:  # 单模态直接使用
            fused_feature = next(iter(embeddings.values()))
        
        # 最终投影与正则化
        output = self.final_proj(self.dropout(fused_feature))
        return F.tanh(output)


class GatedResidualMLPBlock(nn.Module):
    """门控增强版残差块，集成特征选择机制"""
    
    def __init__(self, input_dim: int, expansion: int = 4, dropout: float = 0.2,
                 num_quantiles: Optional[int] = None):
        super().__init__()
        hidden_dim = input_dim * expansion
        
        # 主残差路径
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # 分位数感知门控（可选）
        if num_quantiles:
            self.quant_gate = nn.Linear(num_quantiles, 1, bias=False)
            
        # 门控选择单元
        self.gate_controller = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化门控层权重"""
        nn.init.constant_(self.gate_controller[-2].bias, -2.0)

    def forward(self, x: torch.Tensor, quant_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            quant_emb: 分位数嵌入 (可选)
            
        Returns:
            torch.Tensor: 门控残差输出 [batch_size, input_dim]
        """
        gate = self.gate_controller(x.detach())
        
        # 可选分位数条件门控
        if hasattr(self, 'quant_gate') and quant_emb is not None:
            gate += self.quant_gate(quant_emb)
            
        return x + gate * self.block(x)


class IndependentQuantileHead(nn.Module):
    """独立分位数预测头"""
    
    def __init__(self, hidden_dim: int, quantiles: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(quantiles)
        ])
        
        # 初始化参数
        for head in self.heads:
            # 获取Sequential中的最后一个Linear层
            if isinstance(head, nn.Sequential):
                last_layer = head[-1]
                if isinstance(last_layer, nn.Linear):
                    nn.init.kaiming_normal_(last_layer.weight, mode='fan_in', nonlinearity='linear')
                    nn.init.constant_(last_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征 [batch_size, hidden_dim]
            
        Returns:
            torch.Tensor: 分位数预测 [batch_size, num_quantiles]
        """
        return torch.cat([head(x) for head in self.heads], dim=1)


class StaticPredictor(nn.Module):
    """静态预测器，用于分位数预测或单点预测"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2,
                 num_blocks: int = 3, num_quantiles: int = 3, output_type: str = 'quantile'):
        super().__init__()
        self.output_type = output_type
        self.num_quantiles = num_quantiles
        self.num_blocks = num_blocks
        # 基础特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 堆叠门控残差块
        self.gated_blocks = nn.ModuleList([
            GatedResidualMLPBlock(hidden_dim, expansion=4, dropout=dropout, num_quantiles=(num_quantiles if output_type=='quantile' else None))
            for _ in range(num_blocks)
        ])
        if output_type == 'quantile':
            self.quantile_head = IndependentQuantileHead(hidden_dim, num_quantiles)
        else:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        self.register_buffer('scale', torch.tensor([1e5]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        # 依次通过门控残差块
        for block in self.gated_blocks:
            features = block(features)
        if self.output_type == 'quantile':
            quantiles = self.quantile_head(features)
            return quantiles
        else:
            value = self.value_head(features)
            return value.squeeze(-1)


class MultiScaleSplit(nn.Module):
    """多尺度分割模块，用于提取不同时间尺度的特征"""
    
    def __init__(self, input_dim: int = 1, scales: List[int] = [3, 5, 7], 
                 hidden_dim: int = 64, output_dim: int = 128,
                 constraint_type: str = 'softmax'):
        super().__init__()
        # 强制scales为奇数
        self.scales = [s if s % 2 == 1 else s + 1 for s in scales]
        
        # 卷积层定义
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=s, padding=(s-1)//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for s in scales
        ])
        
        # 可学习的尺度权重
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        self.constraint_type = constraint_type.lower()
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.2),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 多尺度特征 [batch_size, seq_len, output_dim]
        """
        # 输入维度调整
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        
        # 提取多尺度特征
        features = [conv(x).permute(0, 2, 1) for conv in self.conv_layers]
        
        # 应用权重约束
        if self.constraint_type == 'softmax':
            weights = torch.softmax(self.scale_weights, dim=0)
        elif self.constraint_type == 'sigmoid':
            weights = torch.sigmoid(self.scale_weights)
        else:
            raise ValueError(f"不支持的约束类型: {self.constraint_type}")
        
        # 加权融合
        weighted_features = [
            feat * weight.reshape(1, 1, -1) for feat, weight in zip(features, weights)
        ]
        
        # 拼接与融合
        fused = torch.cat(weighted_features, dim=-1)
        return self.fusion(fused)


class DynamicPredictor(nn.Module):
    """动态预测器，用于时间序列预测"""
    
    def __init__(self, input_dim: int, d_model: int, d_n: int, 
                 current_seq_len: int, predict_seq_len: int,
                 num_heads: int = 1, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.current_seq_len = current_seq_len
        self.predict_seq_len = predict_seq_len
        self.d_m = d_model
        self.d_n = d_n
        self.scales = [3, 5, 7]
        
        # 多尺度分割模块
        self.multiscale = MultiScaleSplit(
            input_dim=input_dim,
            scales=self.scales,
            hidden_dim=d_n,
            output_dim=d_model
        )
        
        # 共享的Transformer编码器
        self.shared_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_n * 2,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        # 时间维度转换层
        self.seq_len_adapter = nn.Conv1d(
            in_channels=current_seq_len,
            out_channels=predict_seq_len,
            kernel_size=3,
            padding=1
        )
        
        # 时序演化分支
        self.temporal_proj = nn.Linear(d_model, d_model)
        
        # 跨模态交互分支
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.cross_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        # 自适应融合与输出
        self.fusion_weight = nn.Sequential(
            nn.Linear(d_model, predict_seq_len),
            nn.Sigmoid()
        )
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, seq_input: torch.Tensor, multi_modal: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            seq_input: 时序输入 [batch_size, current_seq_len]
            multi_modal: 多模态特征 [batch_size, d_model]
            
        Returns:
            torch.Tensor: 预测序列 [batch_size, predict_seq_len]
        """
        # 维度调整
        seq_input = seq_input.unsqueeze(-1)  # [batch_size, current_seq_len, 1]
        
        # 多尺度特征提取
        ms_feature = self.multiscale(seq_input)  # [batch_size, current_seq_len, d_model]
        
        # Transformer编码
        encoded = self.shared_encoder(ms_feature.permute(1, 0, 2))  # [current_seq_len, batch_size, d_model]
        encoded = encoded.permute(1, 0, 2)  # [batch_size, current_seq_len, d_model]
        
        # 时序分支
        temporal_feat = self.temporal_proj(encoded)  # [batch_size, current_seq_len, d_model]

        # 跨模态交互分支
        global_context = temporal_feat.mean(dim=1)  # [batch_size, d_model]
        cross_feat = self._cross_processing(global_context, multi_modal)  # [batch_size, predict_seq_len, d_model]

        # 自适应融合
        fused = self._adaptive_fusion(temporal_feat, cross_feat)  # [batch_size, predict_seq_len, d_model]

        return self.decoder(fused).squeeze(-1)

    def _cross_processing(self, global_context: torch.Tensor, 
                         multi_modal: torch.Tensor) -> torch.Tensor:
        """基于全局上下文的跨模态注意力"""
        # 查询向量：全局时序特征
        query = global_context.unsqueeze(0).expand(self.predict_seq_len, -1, -1)
        
        # 键值对：多模态特征
        key = value = multi_modal.unsqueeze(0).expand(self.predict_seq_len, -1, -1)
        
        # 注意力计算
        attn_out, _ = self.cross_attn(query, key, value)
        return self.cross_mlp(attn_out.permute(1, 0, 2))

    def _adaptive_fusion(self, temporal: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        """时间步粒度的动态融合"""
        # 时序特征映射到预测空间
        temporal_proj = self.seq_len_adapter(temporal)  # [batch_size, predict_seq_len, d_model]
        
        # 生成动态融合权重
        attn_weights = torch.matmul(
            temporal_proj, 
            cross.transpose(1, 2)
        ) / np.sqrt(self.d_m)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # 特征融合
        weighted_temporal = torch.matmul(attn_weights, temporal_proj)
        fused = weighted_temporal + cross
        
        return fused
    
    def _generate_position_query(self, device: torch.device, batch_size: int) -> torch.Tensor:
        """生成预测位置查询向量"""
        pos = torch.arange(self.predict_seq_len, device=device).float()
        pos_embed = pos.unsqueeze(1) * nn.Parameter(
            torch.randn(self.predict_seq_len, self.d_m, device=device)
        )
        return pos_embed.unsqueeze(1).expand(-1, batch_size, -1)


class WXM(nn.Module):
    """多模态活动预测模型"""
    
    def __init__(self, args, num_quantiles: int = 3):
        super().__init__()
        self.config = self._create_config_from_args(args)
        self._init_encoders()
        # 新增：静态预测输出类型
        self.static_output_type = getattr(args, 'static_output_type', 'quantile')
        self._init_predictors(num_quantiles)
    
    def _create_config_from_args(self, args) -> ModelConfig:
        """从参数对象创建配置"""
        return ModelConfig(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.d_model,
            d_model=args.d_model,
            d_s=args.d_s,
            d_n=args.d_n,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            current_seq_len=args.current_seq_len,
            predict_seq_len=args.predict_seq_len,
            total_seq_len=args.total_seq_len,
            input_dim=args.input_dim,
            use_img=args.use_img,
            use_text_attributes=args.use_text_attributes,
            use_label_attributes=args.use_label_attributes,
            use_numeric_attributes=args.use_numeric_attributes,
            use_temporal_features=args.use_temporal_features,
            use_current_seq=args.use_current_seq,
            use_similar=args.use_similar,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            dropout=args.dropout,
            gpu_num=args.gpu_num,
            task_type=args.task_type
        )
    
    def _init_encoders(self):
        """初始化各模态编码器"""
        # 文本编码器
        if self.config.use_text_attributes:
            self.text_encoder = TextEmbedder(
                embedding_dim=self.config.embedding_dim,
                gpu_num=self.config.gpu_num,
                dropout=self.config.dropout
            )
        
        # 图像编码器
        if self.config.use_img:
            self.img_encoder = ImageEmbedder()
        
        # 时间特征编码器
        if self.config.use_temporal_features:
            self.temporal_encoder = DummyEmbedder(
                embedding_dim=self.config.embedding_dim,
                dropout=self.config.dropout
            )
        
        # 多模态融合模块
        self.multimodal_fusion = CombinedModel(
            use_img=self.config.use_img,
            use_text_attributes=self.config.use_text_attributes,
            use_label_attributes=self.config.use_label_attributes,
            use_numeric_attributes=self.config.use_numeric_attributes,
            use_temporal_features=self.config.use_temporal_features,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.d_model,
            num_heads=self.config.num_heads,
            gpu_num=self.config.gpu_num,
            dropout=self.config.dropout
        )
    
    def _init_predictors(self, num_quantiles: int):
        # 静态预测器
        if self.static_output_type == 'quantile':
            self.static_predictor = StaticPredictor(
                input_dim=self.config.d_model,
                hidden_dim=self.config.d_s,
                dropout=self.config.dropout,
                num_blocks=getattr(self.config, 'num_gated_blocks', 2),
                num_quantiles=num_quantiles,
                output_type='quantile'
            )
        else:
            self.static_predictor = StaticPredictor(
                input_dim=self.config.d_model,
                hidden_dim=self.config.d_s,
                dropout=self.config.dropout,
                num_blocks=getattr(self.config, 'num_gated_blocks', 2),
                num_quantiles=1,
                output_type='value'
            )
        # 动态预测器
        self.dynamic_predictor = DynamicPredictor(
            input_dim=self.config.input_dim,
            d_model=self.config.d_model,
            d_n=self.config.d_n,
            current_seq_len=self.config.current_seq_len,
            predict_seq_len=self.config.predict_seq_len,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
    
    def forward(self, input_sequence: torch.Tensor, numeric_features: torch.Tensor,
                label_features: torch.Tensor, temporal_features: torch.Tensor,
                activity_text: Dict[str, List[str]], 
                similar_sequences: Optional[torch.Tensor] = None) -> torch.Tensor:
        multimodal_features = self._encode_multimodal_features(
            numeric_features, label_features, temporal_features, activity_text
        )
        if self.config.task_type == 'static':
            return self.static_predictor(multimodal_features)
        else:
            return self.dynamic_predictor(input_sequence, multimodal_features)
    
    def _encode_multimodal_features(self, numeric_features: torch.Tensor,
                                  label_features: torch.Tensor,
                                  temporal_features: torch.Tensor,
                                  activity_text: Dict[str, List[str]]) -> torch.Tensor:
        """编码多模态特征"""
        # 时间特征编码
        temporal_embeddings = None
        if self.config.use_temporal_features:
            temporal_embeddings = self.temporal_encoder(temporal_features)
        
        # 文本特征编码
        text_embeddings = None
        if self.config.use_text_attributes:
            text_embeddings = self.text_encoder(
                activity_text['activity_name'],
                activity_text['activity_title'],
                activity_text['product_names']
            )
        
        # 多模态融合
        return self.multimodal_fusion(
            numeric_features, label_features, text_embeddings, temporal_embeddings
        )

