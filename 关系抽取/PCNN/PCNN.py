import torch
import torch.nn as nn


class PCNN(nn.Module):

    def __init__(self, hidden_dim, kernel_size, num_labels, dropout=0.5):
        super(PCNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # 卷积层
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size)

        # 池化层
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        # dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        # x的形状：[batch_size, seq_length, hidden_dim]
        x = x.permute(0, 2, 1)  # 转换成：[batch_size, hidden_dim, seq_length]
        x = x.to(torch.float32)
        # 卷积操作
        x = self.conv(x)  # 卷积后：[batch_size, hidden_dim, seq_length - kernel_size + 1]

        # 池化操作：分别进行最大池化和平均池化
        max_pool_out = self.max_pool(x)
        avg_pool_out = self.avg_pool(x)

        # 将池化的结果连接起来
        pooled_out = torch.cat((max_pool_out, avg_pool_out), dim=1)  # [batch_size, hidden_dim * 2, pooled_seq_len]

        # flatten成一维
        pooled_out = pooled_out.view(pooled_out.size(0), -1)

        # dropout层
        pooled_out = self.dropout(pooled_out)

        # 全连接层
        out = self.fc(pooled_out)

        return out
