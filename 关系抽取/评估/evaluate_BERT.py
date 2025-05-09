import os

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import logging
from transformers import BertTokenizer, AdamW, BertConfig
import utils
from 关系抽取任务.BERT.dataset import NewsDataset
from torch.utils.data import DataLoader
from sklearn import metrics
from 关系抽取任务.BERT.model import BertClassifier


def train():
    #需修改的三条
    model_name = 'RE_BERT.pkl'
    log_dir = 'logs/eva_RE_BERT.log'
    test_path = 'eva_data4.json'



    utils.set_logger(log_dir)
    model_path = r'/home/ypx/project/ypx/Agriculture_projects/models/bert-base-chinese'
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    learning_rate = 5e-6
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 获取到dataset
    label_map = {"属于": 0, "防治": 1, "感染": 2, "别名": 3, "被危害": 4, "科": 5, "导致": 6, "病原": 7, "目": 8,
                 "纲": 9, "不良反应": 10}
    # "属于": 0, "防治": 1, "感染": 2, "别名": 3, "被危害": 4, "科":5, "导致":6, "病原":7, "目":8, "纲":9, "不良反应":10
    test_dataset = NewsDataset(test_path, tokenizer, label_map)
    logging.info('Loaded datasets.')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    bert_config = BertConfig.from_pretrained(model_path)
    num_labels = len(label_map)
    logging.info('testing model: {0}'.format(model_name))
    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)
    model.load_state_dict(torch.load('../finetuning_models/{0}'.format(model_name)))
    # 优化器
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    model.eval()
    losses = 0  # 损失
    pred_labels = []
    true_labels = []
    test_bar = tqdm(test_dataloader, ncols=100)
    for inputs, labels in test_bar:
        input_ids = inputs['input_ids'].squeeze(1)
        attention_mask = inputs['attention_mask'].squeeze(1)
        token_type_ids = inputs['token_type_ids'].squeeze(1)
        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            token_type_ids=token_type_ids.to(device),
        )

        loss = criterion(output, labels.to(device))
        losses += loss.item()

        pred_label = torch.argmax(output, dim=1)  # 预测出的label
        acc = torch.sum(pred_label == labels.to(device)).item() / len(pred_label)  # acc
        test_bar.set_postfix(loss=loss.item(), acc=acc)

        pred_labels.extend(pred_label.cpu().numpy().tolist())
        true_labels.extend(labels.cpu().numpy().tolist())

    average_loss = losses / len(test_dataloader)
    # print('\tLoss:', average_loss)
    logging.info('\ttest Loss: {0}'.format(average_loss))
    # 分类报告
    report = metrics.classification_report(true_labels, pred_labels, digits=5)
    # print('* Classification Report:')
    # print(report)
    logging.info('*{0} test Classification Report:'.format(model_name))
    logging.info('\n' + report)

if __name__ == '__main__':
    train()
