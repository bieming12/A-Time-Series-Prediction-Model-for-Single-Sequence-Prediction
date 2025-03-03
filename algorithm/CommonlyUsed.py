import torch
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self, input_size, sequence_length, output_size, num):
        super().__init__()
        # 定义全连接层的结构
        self.fc1 = nn.Linear(input_size * sequence_length, num)
        self.fc2 = nn.Linear(num, output_size)
        # 添加激活函数层
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        # 输入序列的形状是 [batch_size, input_size, sequence_length]
        # 我们需要将输入展平以适应全连接层
        batch_size, input_size, sequence_length = input_seq.size()
        flat_input = input_seq.view(batch_size, -1)  # 自动计算展平后的维度
        x = self.fc1(flat_input)
        x = self.relu(x)
        output = self.fc2(x)
        return output

# GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gru1 = nn.GRU(input_size, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 16, batch_first=True)
        self.linear = nn.Linear(16, output_size)
    def forward(self, input_seq):
        # x: [Batch, Channel, Input length]
        # 转换形状以匹配 decomposition 的期望输入形状 [Batch, Input length, Channel]
        x_permuted = input_seq.permute(0, 2, 1)
        lstm_out1, _ = self.gru1(x_permuted)
        lstm_out2, _ = self.gru2(lstm_out1)
        output = self.linear(lstm_out2[:, -1, :])
        output = output.squeeze()
        return output

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.linear = nn.Linear(64, output_size)
    def forward(self, input_seq):
        # x: [Batch, Channel, Input length]
        # 转换形状以匹配 decomposition 的期望输入形状 [Batch, Input length, Channel]
        x_permuted = input_seq.permute(0, 2, 1)
        lstm_out1, _ = self.lstm1(x_permuted)
        lstm_out2, _ = self.lstm2(lstm_out1)
        output = self.linear(lstm_out2[:, -1, :])
        output = output.squeeze()
        return output