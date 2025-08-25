import torch
import os
from models.WXM import WXM
from train import get_args  # 导入 train.py 中的 get_args 函数

# 使用训练好的模型预测新活动
def load_model(model_class, model_path, device, **model_kwargs):
    """
    加载训练好的模型
    :param model_class: 模型类
    :param model_path: 模型文件路径
    :param device: 设备（'cpu' 或 'cuda'）
    :param model_kwargs: 初始化模型所需的其他参数
    :return: 加载好的模型
    """
    # 初始化模型架构
    model = model_class(**model_kwargs).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置模型为评估模式
    return model

def predict(model, data_loader, device):
    """
    使用模型进行预测
    :param model: 加载好的模型
    :param data_loader: 数据加载器
    :param device: 设备（'cpu' 或 'cuda'）
    :return: 预测结果
    """
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
    return predictions

# 示例用法
if __name__ == "__main__":
    # 从 train.py 中获取参数
    args = get_args()
    args.model_path = 'models/model.pth'  # 设置模型路径

    # 设置设备
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = load_model(WXM, args.model_path, device,
                       embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim,
                       num_heads=args.num_attn_heads,
                       num_layers=args.num_hidden_layers,
                       use_img=args.use_img,
                       use_attributes=args.use_attributes,
                       use_temporal_features=args.use_temporal_features,
                       use_current_seq=args.use_current_seq,
                       current_seq_len=args.current_seq_len,
                       predict_seq_len=args.predict_seq_len,
                       num_features=args.num_features,
                       gpu_num=args.gpu_num)
    
    # 创建数据加载器
    # 假设你已经有一个 DataLoader 对象 new_activity_loader
    new_activity_loader = ...

    # 使用模型进行预测
    predictions = predict(model, new_activity_loader, device)
    print(predictions)