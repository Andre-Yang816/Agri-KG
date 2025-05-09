import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertTextCNNClassifier(nn.Module):
    def __init__(self, bert_config, num_labels, kernel_sizes=[2, 3, 4], num_filters=128):
        super().__init__()
        self.bert = BertModel(config=bert_config)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, bert_config.hidden_size))
            for k in kernel_sizes
        ])
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state.unsqueeze(1)  # (batch, 1, seq_len, hidden_size)

        conv_outs = [F.relu(conv(last_hidden_state)).squeeze(3) for conv in
                     self.convs]  # (batch, num_filters, seq_len-k+1)
        pooled_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]  # (batch, num_filters)

        concat_out = torch.cat(pooled_outs, dim=1)  # (batch, num_filters * len(kernel_sizes))
        logits = self.classifier(concat_out)

        return logits
