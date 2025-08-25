import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from models.WXM import WXM
import random
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        
    def forward(self, preds, target):
        """
        preds: [batch_size, num_quantiles] 
        target: [batch_size]
        """
        # 显式扩展target维度
        target = target.repeat_interleave(len(self.quantiles)).view(-1, len(self.quantiles))  # [B, Q]
        
        # 逐元素计算误差
        errors = target - preds  # [B, Q]
        
        # 分位数损失计算
        losses = []
        for i, q in enumerate(self.quantiles):
            q_tensor = torch.full_like(errors[:, i], q)  # 显式分位数参数
            loss = torch.max( (q_tensor-1)*errors[:, i], q_tensor*errors[:, i] )
            losses.append(loss)
        
        # 合并损失并平均
        total_loss = torch.stack(losses, dim=1).mean()  # [B,Q] → scalar
        return total_loss

class StaticExp:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
        self.results_dir = 'results/static'
        self.results_file = os.path.join(self.results_dir, 'static_results.txt')
        self.model_save_dir = 'models'
        self.model_save_path = os.path.join(self.model_save_dir, f"{args.model_type}_{args.task_type}.pth")
        
        # 新增：根据参数决定分位数或单点预测
        self.static_output_type = getattr(args, 'static_output_type', 'quantile')
        if self.static_output_type == 'quantile':
            self.quantiles = [0.1, 0.5, 0.9]
            self.model = WXM(args, num_quantiles=len(self.quantiles)).to(self.device)
            self.criterion = self.get_loss_function("quantile")
        else:
            self.quantiles = None
            self.model = WXM(args, num_quantiles=1).to(self.device)
            self.criterion = self.get_loss_function(args.loss)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.static_learning_rate)
        
        # 学习率调度器
        if args.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', patience=3, factor=0.5
            )

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_loss_function(self, loss_name):
        if loss_name == 'quantile':
            return QuantileLoss([0.1, 0.5, 0.9])
        elif loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def train(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            inputs, target = self._prepare_batch(batch)
            
            # 前向传播
            outputs = self.model(**inputs)
            #print(outputs)
            loss = self.criterion(outputs, target)
            
            # 反向传播
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * target.size(0)
            
        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # 数据加载与训练相同
                inputs, target = self._prepare_batch(batch)
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, target)
                total_loss += loss.item() * target.size(0)
                
        return total_loss / len(val_loader.dataset)

    def test(self, test_loader):
        self.model.eval()
        if self.static_output_type == 'quantile':
            predictions = {f'q{int(q*100)}': [] for q in [0.1, 0.5, 0.9]}
        else:
            predictions = {'value': []}
        true_values = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, target = self._prepare_batch(batch)
                outputs = self.model(**inputs)
                if self.static_output_type == 'quantile':
                    for i, q in enumerate([0.1, 0.5, 0.9]):
                        key = f'q{int(q*100)}'
                        predictions[key].extend(outputs[:,i].cpu().numpy())
                else:
                    predictions['value'].extend(outputs.cpu().numpy().flatten())
                true_values.extend(target.cpu().numpy())
        
        return predictions, np.array(true_values)

    def _prepare_batch(self, batch):
        """ 统一处理批次数据 """
        (input_sequence, target_sequence, target, 
         numeric_features, label_features, 
         temporal_features, activity_text, 
         similar_sequences) = batch
        
        inputs = {
            'input_sequence': input_sequence.to(self.device),
            'numeric_features': numeric_features.to(self.device),
            'label_features': label_features.to(self.device),
            'temporal_features': temporal_features.to(self.device),
            'activity_text': activity_text,
            'similar_sequences': similar_sequences.to(self.device)
        }
        target = target.to(self.device)
        return inputs, target

    def plot_results(self, true_values, predictions, num_samples=500):
        """ 分位数预测或单点预测可视化，风格与dynamic_exp一致 """
        import matplotlib
        plt.rcParams['font.family'] = 'Times New Roman'
        output_dir = os.path.join(
            self.results_dir,
            f"{self.args.model_type}_epoch{self.args.static_epochs:03d}_seed{self.args.random_seed}"
        )
        os.makedirs(output_dir, exist_ok=True)
        true_values = np.asarray(true_values).flatten()
        if self.static_output_type == 'quantile':
            pred_dict = {
                'q10': np.asarray(predictions['q10']).flatten(),
                'q50': np.asarray(predictions['q50']).flatten(),
                'q90': np.asarray(predictions['q90']).flatten()
            }
        else:
            pred_dict = {'value': np.asarray(predictions['value']).flatten()}
        max_samples = len(true_values)
        num_samples = min(num_samples, max_samples) if num_samples else max_samples
        idx = np.random.choice(len(true_values), num_samples, replace=False)
        tv_sample = true_values[idx]
        x_axis = np.arange(num_samples)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.set_facecolor('white')
        if self.static_output_type == 'quantile':
            q10_sample = pred_dict['q10'][idx]
            q50_sample = pred_dict['q50'][idx]
            q90_sample = pred_dict['q90'][idx]
            # 置信区间
            plt.fill_between(
                x_axis, 
                q10_sample,
                q90_sample,
                color='#B7E2F0',  # 更淡的蓝色
                alpha=0.4,
                label='80% Confidence Interval'
            )
            plt.plot(x_axis, q10_sample, color='#2C91ED', linewidth=2, linestyle=':', label='10th Percentile')
            plt.plot(x_axis, q50_sample, color='#F0A73A', linewidth=2, label='Median Prediction')
            plt.plot(x_axis, q90_sample, color='#3ABF99', linewidth=2, linestyle='--', label='90th Percentile')
            plt.plot(x_axis, tv_sample, color='#3A3A3A', linewidth=2, linestyle='-', label='True Values')
            coverage = np.mean((tv_sample >= q10_sample) & (tv_sample <= q90_sample))
            mae_q50 = mean_absolute_error(tv_sample, q50_sample)
            interval_width = np.mean(q90_sample - q10_sample)
            stats_text = (
                f'Coverage Rate: {coverage:.1%}\n'
                f'Median MAE: {mae_q50:.2f}\n'
                f'Average Interval Width: {interval_width:.2f}'
            )
            plt.text(0.70, 0.15, stats_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.legend(loc='upper right', fontsize=12)
            plt.title('Quantile Regression Prediction Analysis', fontsize=16)
            np.save(os.path.join(output_dir, 'true_values.npy'), true_values)
            for q_key in ['q10', 'q50', 'q90']:
                np.save(os.path.join(output_dir, f'{q_key}_predictions.npy'), pred_dict[q_key])
        else:
            value_sample = pred_dict['value'][idx]
            plt.plot(x_axis, value_sample, color='#F0A73A', linewidth=2, label='Prediction')
            plt.plot(x_axis, tv_sample, color='#3ABF99', linewidth=2, linestyle='-', label='True Values')
            mae = mean_absolute_error(tv_sample, value_sample)
            stats_text = f'MAE: {mae:.2f}'
            plt.text(0.70, 0.15, stats_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.legend(loc='upper right', fontsize=12)
            plt.title('Value Prediction Analysis', fontsize=16)
            np.save(os.path.join(output_dir, 'true_values.npy'), true_values)
            np.save(os.path.join(output_dir, 'value_predictions.npy'), pred_dict['value'])
        plt.xlabel('Sample Index', fontsize=14)
        plt.ylabel('Target Value', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(os.path.join(output_dir, 'quantile_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def run(self, train_loader, test_loader):
        self.set_seed(self.args.random_seed)
        best_val_loss = float('inf')
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        num_epochs = self.args.static_epochs
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = self.train(train_loader)
            
            # 验证阶段
            val_loss = self.validate(test_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
            
            
            # 打印日志
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            # 更新学习率调度器
            if self.args.use_lr_scheduler:
                self.scheduler.step(val_loss)  # 使用 val_loss 作为监控指标
                # self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}, Updated Learning Rate: {current_lr:.6f}')
            print('-' * 50)
        
        # 最终测试
        self.model.load_state_dict(torch.load(self.model_save_path))
        predictions, true_values = self.test(test_loader)
        self.plot_results(true_values, predictions)
        
        # 保存预测结果
        with open(self.results_file, 'a') as f:
            f.write(f'Epochs {num_epochs},  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')
        
        return predictions, true_values