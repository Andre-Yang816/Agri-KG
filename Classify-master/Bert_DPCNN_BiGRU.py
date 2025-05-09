import torch
import torch.nn as nn
from transformers import BertModel


class BertDPCNNBiGRU(nn.Module):
    def __init__(self, bert_config, num_labels, num_filters=128):
        super().__init__()
        self.bert = BertModel(config=bert_config)

        # DPCNN 部分
        self.conv_region = nn.Conv2d(1, num_filters, (3, bert_config.hidden_size), stride=1, padding=(1, 0))
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1, padding=(1, 0))
        self.downsample = nn.Conv2d(num_filters, num_filters, (3, 1), stride=2, padding=(1, 0))  # 可能导致降维

        self.relu = nn.ReLU()

        # BiGRU 部分
        self.gru = nn.GRU(num_filters, num_filters, bidirectional=True, batch_first=True)

        # 分类层
        self.fc = nn.Linear(num_filters * 2, num_labels)

    def _block(self, x):
        """DPCNN 残差块"""
        residual = x  # 残差连接

        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.relu(x)

        # 使残差尺寸与 downsample(x) 保持一致
        if residual.shape != x.shape:
            residual = self.downsample(residual)  # 降维
        x = x + residual  # 残差连接
        return x

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = bert_output.last_hidden_state  # (batch, seq_len, hidden_dim)

        x = x.unsqueeze(1)  # 添加 channel 维度, 变成 (batch, 1, seq_len, hidden_dim)

        x = self.conv_region(x)  # 区域卷积
        x = self.relu(x)

        x = self._block(x)
        x = self._block(x)  # 多次 DPCNN 残差块

        x = x.squeeze(3)  # 去掉最后一维 (batch, num_filters, seq_len)
        x = x.permute(0, 2, 1)  # 变成 (batch, seq_len, num_filters)，适配 GRU

        x, _ = self.gru(x)  # 经过 BiGRU 处理

        x = self.fc(x[:, -1, :])  # 取最后时间步进行分类
        return x
