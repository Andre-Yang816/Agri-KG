
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from transformers import AutoModel


class ERNIE_BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(ERNIE_BiLSTM_CRF, self).__init__()
        self.num_labels = config.num_labels
        self.ernie = AutoModel.from_pretrained(config.ernie_model_path)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.lstm_embedding_size,
            hidden_size=config.hidden_size,  # 512
            batch_first=True,
            num_layers=2,
            dropout=config.dropout_prob,  # 0.5
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.ernie(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds
                            )
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
