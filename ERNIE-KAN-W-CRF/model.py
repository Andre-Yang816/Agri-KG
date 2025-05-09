from kan import KANLayer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.special import logit
from torchcrf import CRF
from transformers import AutoModel

from kg_classify_BERT.config import hidden_size


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
        self.kan_layer = KANLayer(in_dim = config.hidden_size*2, out_dim = config.num_labels, device=config.device)

        self.crf = CRF(config.num_labels, batch_first=True)
        self.label_weights = config.class_weights
        self.device = config.device

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
        lstm_output = padded_sequence_output
        batch_size, seq_len, hidden_dim = lstm_output.shape
        lstm_output_flat = lstm_output.reshape(-1, hidden_dim)
        logits, _, _, _ = self.kan_layer(lstm_output_flat)
        logits = logits.view(batch_size, seq_len, -1)

        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            self.label_weights = self.label_weights.to(self.device)
            weights = self.label_weights[labels]
            loss = self.crf(logits, labels, loss_mask) * (-1)
            weighted_loss = (weights * loss).mean()
            outputs = (weighted_loss,) + outputs

        # contain: (loss), scores
        return outputs
