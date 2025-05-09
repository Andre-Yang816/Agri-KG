import torch
import torch.nn as nn
from transformers import BertModel


class SharedBERTWithAttention(nn.Module):
    def __init__(self, bert_config, num_classes=4):
        super(SharedBERTWithAttention, self).__init__()
        self.bert = BertModel(config=bert_config)
        self.attention = nn.Linear(768 * 3, 3)  # 用于计算标题、实体、句子的权重
        self.fc = nn.Linear(768, num_classes)  # 分类层
        self.dropout = nn.Dropout(0.3)

    def forward(self, title_input, entity_input, sentence_input, gra_input):
        # 共享 BERT，分别处理不同输入

        title_input_ids = title_input['input_ids'].squeeze(1)
        entity_input_ids = entity_input['input_ids'].squeeze(1)
        sentence_input_ids = sentence_input['input_ids'].squeeze(1)
        gra_input_ids = gra_input['input_ids'].squeeze(1)
        output1 = self.bert(input_ids=title_input_ids, attention_mask=title_input['attention_mask'].squeeze(1),
                            token_type_ids=title_input['token_type_ids'].squeeze(1))
        output2 = self.bert(input_ids=entity_input_ids, attention_mask=entity_input['attention_mask'].squeeze(1),
                            token_type_ids=entity_input['token_type_ids'].squeeze(1))
        output3 = self.bert(input_ids=sentence_input_ids, attention_mask=sentence_input['attention_mask'].squeeze(1),
                            token_type_ids=sentence_input['token_type_ids'].squeeze(1))
        output4 = self.bert(input_ids=gra_input_ids, attention_mask=gra_input['attention_mask'].squeeze(1),
                            token_type_ids=gra_input['token_type_ids'].squeeze(1))

        h1 = output1.last_hidden_state[:, 0, :]  # 标题的 [CLS]
        h2 = output2.last_hidden_state[:, 0, :]  # 实体的 [CLS]
        h3 = output3.last_hidden_state[:, 0, :]  # 句子的 [CLS]
        h4 = output4.last_hidden_state[:, 0, :]  # 图谱的 [CLS]
        # 计算注意力权重
        weights = torch.softmax(self.attention(torch.cat([h1, h2, h3, h4], dim=-1)), dim=-1)

        # 计算加权和
        fused_vector = weights[:, 0:1] * h1 + weights[:, 1:2] * h2 + weights[:, 2:3] * h3 + weights[:, 3:4] * h4
        fused_vector = self.dropout(fused_vector)

        # 获取 input_ids、attention_mask 等

        # 分类
        output = self.fc(fused_vector)
        return output
