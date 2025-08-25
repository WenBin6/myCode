import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.WXM import WXM
from models.DDN import DDN
import numpy as np
import os


class DynamicExp:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
        self.results_dir = 'results/dynamic'
        self.results_file = os.path.join(self.results_dir, 'dynamic_results.txt')
        self.model_save_dir = 'models'
        self.model_save_path = os.path.join(self.model_save_dir, f"{args.model_type}_{args.task_type}.pth")

        # 初始化模型
        self.ddn_model = DDN(args).to(self.device) if args.use_ddn_normalization else None
        self.model = WXM(args).to(self.device)

        # 初始化损失函数和优化器
        self.criterion = self.get_loss_function(args.loss)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.dynamic_learning_rate)
        self.station_optim = optim.Adam(self.ddn_model.parameters(), lr=args.station_lr) if args.use_ddn_normalization else None
        # 学习率调度器
        if args.use_lr_scheduler:
            try:
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,  # 需要调度的优化器
                    mode='min',      # 监控的指标需要最小化（如 loss）
                    factor=0.5,      # 学习率降低的因子（new_lr = lr * factor）
                    patience=5,      # 等待 5 个 epoch，如果指标没有改善，则降低学习率
                    verbose=True,    # 打印学习率更新的信息
                    threshold=1e-4,  # 指标变化的阈值，只有变化超过阈值才认为是改善
                    threshold_mode='rel',  # 使用相对变化（'rel'）或绝对变化（'abs'）
                    cooldown=0,      # 降低学习率后的冷却时间（epoch 数）
                    min_lr=1e-6      # 学习率的下限
                )
            except TypeError:
                # 如果verbose参数不被支持，使用简化版本
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    threshold=1e-4,
                    threshold_mode='rel',
                    cooldown=0,
                    min_lr=1e-6
                )

    @staticmethod
    def get_loss_function(loss_name):
        if loss_name == 'mse':
            return nn.MSELoss(reduction='mean')
        elif loss_name == 'mae':
            return nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def station_loss(self, y, statistics_pred):
        # 输入y 的形状为 [b, pre_len]
        # statistics_pred 的形状为 [b, pre_len, 2],2个维度分别是标准差和均值
        y = y.unsqueeze(-1)
       
        bs, len, dim= y.shape
        # mean [bs,pre_len]
        mean = torch.mean(y, dim=2)
        # 手动计算标准差
        # 将 mean 的形状扩展为与 y 匹配
        mean_expanded = mean.unsqueeze(-1)
        var = torch.mean((y - mean_expanded) ** 2, dim=2)  # 计算方差
        # std [bs,pre_len]
        std = torch.sqrt(var + 1e-7)                       # 计算标准差，添加微小常数
        #求出来的std值为nan
        #std = torch.std(y, dim=2)
        # 将 mean 和 std 拼接为 [b, pre_len, 2]
        station_true = torch.stack([mean, std], dim=-1)  # 使用 stack 而不是 cat
        #station_ture = torch.cat([mean, std], dim=-1)
        #print(station_true.shape)
        # 调整 station_ture 的形状为 [bs, pre_len, 2]
        #station_ture = station_ture.view(bs, self.args.predict_seq_len, 2)
        loss = self.criterion(statistics_pred, station_true)
        return loss
    def train(self, train_loader, epoch, station_pretrain_epoch):
        self.model.train()
        if self.args.use_ddn_normalization:
            self.ddn_model.train()
        train_loss = 0.0

        for i, batch in enumerate(train_loader):
            try:
                self.optimizer.zero_grad()
                if self.args.use_ddn_normalization:
                    self.station_optim.zero_grad()

                input_sequence, target_sequence, target, numeric_features, label_features, temporal_features, activity_text, similar_sequences = batch
                input_sequence = input_sequence.to(self.device)
                target_sequence = target_sequence.to(self.device)
                numeric_features = numeric_features.to(self.device)
                label_features = label_features.to(self.device)
                temporal_features = temporal_features.to(self.device)
                similar_sequences = similar_sequences.to(self.device)

                # 将 input_sequence 从 [batch_size, seq_len] 扩展为 [batch_size, seq_len, 1]
                input_sequence = input_sequence.unsqueeze(-1)

                # DDN 归一化
                if self.args.use_ddn_normalization and epoch + 1 <= station_pretrain_epoch:
                    input_sequence, statistics_pred, _ = self.ddn_model.normalize(input_sequence)
                    loss = self.station_loss(target_sequence, statistics_pred)
                else:
                    if self.args.use_ddn_normalization:
                        input_sequence, statistics_pred, _ = self.ddn_model.normalize(input_sequence)
                        #print("input_sequence shape",input_sequence.shape)
                        output = self.model(input_sequence.squeeze(-1), numeric_features, label_features, temporal_features, activity_text, similar_sequences)
                        #print("output shape",output.shape)
                        output = self.ddn_model.de_normalize(output.unsqueeze(-1), statistics_pred)
                        #print("output shape",output.shape)
                    else:
                        #模型output变成nan了
                        output = self.model(input_sequence.squeeze(-1), numeric_features, label_features, temporal_features, activity_text, similar_sequences)
                    # output[B, seq_len, 1] -> [B, seq_len]
                    loss = self.criterion(output.squeeze(-1), target_sequence)

                # 反向传播和参数更新
                loss.backward()
                if self.args.use_ddn_normalization and epoch + 1 <= station_pretrain_epoch:
                    self.station_optim.step()
                else:
                    self.optimizer.step()
                    if self.args.use_ddn_normalization:
                        self.station_optim.step()

                train_loss += loss.item() * target_sequence.size(0)

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                raise

        return train_loss / len(train_loader.dataset)

    def validate(self, val_loader, epoch, station_pretrain_epoch):
        self.model.eval()
        if self.args.use_ddn_normalization:
            self.ddn_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_sequence, target_sequence, target, numeric_features, label_features, temporal_features, activity_text, similar_sequences = batch
                input_sequence = input_sequence.to(self.device)
                target_sequence = target_sequence.to(self.device)
                numeric_features = numeric_features.to(self.device)
                label_features = label_features.to(self.device)
                temporal_features = temporal_features.to(self.device)
                similar_sequences = similar_sequences.to(self.device)

                input_sequence = input_sequence.unsqueeze(-1)

                if self.args.use_ddn_normalization and epoch + 1 <= station_pretrain_epoch:
                    input_sequence, statistics_pred, _ = self.ddn_model.normalize(input_sequence, p_value=False)
                    loss = self.station_loss(target_sequence, statistics_pred)
                else:
                    if self.args.use_ddn_normalization:
                        input_sequence, statistics_pred, _ = self.ddn_model.normalize(input_sequence)
                        output = self.model(input_sequence.squeeze(-1), numeric_features, label_features, temporal_features, activity_text, similar_sequences)
                        output = self.ddn_model.de_normalize(output.unsqueeze(-1), statistics_pred)
                    else:
                        output = self.model(input_sequence.squeeze(-1), numeric_features, label_features, temporal_features, activity_text, similar_sequences)
                    loss = self.criterion(output.squeeze(-1), target_sequence)

                val_loss += loss.item() * target_sequence.size(0)

        torch.cuda.empty_cache()
        return val_loss / len(val_loader.dataset)

    def test(self, test_loader):
        self.model.eval()
        if self.args.use_ddn_normalization:
            self.ddn_model.eval()
        predictions = []
        true_values = []
        current_values_list = []

        with torch.no_grad():
            for batch in test_loader:
                input_sequence, target_sequence, target, numeric_features, label_features, temporal_features, activity_text, similar_sequences = batch
                input_sequence = input_sequence.to(self.device)
                target_sequence = target_sequence.to(self.device)
                numeric_features = numeric_features.to(self.device)
                label_features = label_features.to(self.device)
                temporal_features = temporal_features.to(self.device)
                similar_sequences = similar_sequences.to(self.device)

                input_sequence = input_sequence.unsqueeze(-1)
                #注意：需要在ddn归一化前保存输入序列真实值，不然可能会出现负数
                current_values_list.append(input_sequence.cpu().numpy())
                if self.args.use_ddn_normalization:
                    input_sequence, statistics_pred, _ = self.ddn_model.normalize(input_sequence)
                    output = self.model(input_sequence.squeeze(-1), numeric_features, label_features, temporal_features, activity_text, similar_sequences)
                    output = self.ddn_model.de_normalize(output.unsqueeze(-1), statistics_pred)
                else:
                    output = self.model(input_sequence.squeeze(-1), numeric_features, label_features, temporal_features, activity_text, similar_sequences)
                output = output.squeeze(-1)

                predictions.append(output.cpu().numpy())
                true_values.append(target_sequence.cpu().numpy())

        # Predictions shape: (123, 7)
        # True values shape: (123, 7)
        # Current values shape: (123, 14, 1)
        predictions = np.concatenate(predictions, axis=0)
        true_values = np.concatenate(true_values, axis=0)
        current_values = np.concatenate(current_values_list, axis=0)
        print(f"Predictions shape: {predictions.shape}")
        print(f"True values shape: {true_values.shape}")
        print(f"Current values shape: {current_values.shape}")
        return predictions, true_values, current_values

    def plot_results(self, true_values, predictions, current_values):
        plt.rcParams['font.family'] = 'Times New Roman'

        filename = f"{self.args.model_type}_epoch{self.args.dynamic_epochs}_{self.args.current_seq_len}_{self.args.predict_seq_len}_{self.args.total_seq_len}_{self.args.use_ddn_normalization}_seed{self.args.random_seed}"
        output_dir = os.path.join(self.results_dir, filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(len(current_values)):
            plt.figure(figsize=(10, 6))
            con_true_values = np.concatenate([current_values[i].flatten(), true_values[i].flatten()])
            con_predictions = np.concatenate([current_values[i].flatten(), predictions[i].flatten()])
            plt.plot(range(self.args.current_seq_len + self.args.predict_seq_len), con_true_values, label='Ground Truth', color='#3ABF99', linewidth=2)
            plt.plot(range(self.args.current_seq_len + self.args.predict_seq_len), con_predictions, label='Prediction', color='#F0A73A', linewidth=2)
            plt.plot(range(self.args.current_seq_len), current_values[i].flatten(), label='Input Data', color='#2C91ED', linewidth=2)
            plt.legend(loc="upper right")
            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel('OC', fontsize=14)
            plt.title('Predictions vs True Values', fontsize=16)
            plt.legend(fontsize=12)
            plt.gca().set_facecolor('white')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(os.path.join(output_dir, f"{i+1}.png"))
            plt.close()

        
        np.save(os.path.join(output_dir,"_predictions.npy"), predictions)
        np.save(os.path.join(output_dir,"_true_values.npy"), true_values)

    def run(self, train_loader, test_loader):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        num_epochs = self.args.dynamic_epochs
        station_pretrain_epoch = self.args.pre_epoch

        for epoch in range(num_epochs + station_pretrain_epoch):
            train_loss = self.train(train_loader, epoch, station_pretrain_epoch)
            val_loss = self.validate(test_loader, epoch, station_pretrain_epoch)
            if epoch + 1 <= station_pretrain_epoch:
                print(f'Station Pretrain Epoch {epoch+1}/{station_pretrain_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            # 更新学习率调度器
            if self.args.use_lr_scheduler:
                self.scheduler.step(val_loss)  # 使用 val_loss 作为监控指标
                # self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}, Updated Learning Rate: {current_lr:.6f}')
            print('-' * 50)

        predictions, true_values, current_values = self.test(test_loader)
        mae = mean_absolute_error(true_values, predictions)
        print(f'Mean Absolute Error: {mae:.4f}')
        mse = mean_squared_error(true_values, predictions)
        print(f'Mean Squared Error: {mse:.4f}')
        mape = mean_absolute_percentage_error(true_values, predictions)
        print(f'Mean Absolute Percentage Error: {mape:.4f}')
        self.plot_results(true_values, predictions, current_values)
        
        with open(self.results_file, 'a') as f:
            f.write(f'Epochs {num_epochs}, MAE: {mae:.4f}, MSE:{mse:.4f}, MAPE:{mape:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},current_seq_len: {self.args.current_seq_len},predict_seq_len: {self.args.predict_seq_len},total_seq_len: {self.args.total_seq_len}\n')

        # 保存WXM模型
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f'WXM Model saved to {self.model_save_path}')
        
        # 如果使用DDN，也保存DDN模型参数
        if self.args.use_ddn_normalization and self.ddn_model is not None:
            ddn_save_path = os.path.join(self.model_save_dir, f"DDN_{self.args.task_type}.pth")
            torch.save(self.ddn_model.state_dict(), ddn_save_path)
            print(f'DDN Model saved to {ddn_save_path}')
        
        return predictions, true_values