import os

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import logging
from transformers import BertTokenizer, AdamW, BertConfig
import utils
from dataset import NewsDataset
from torch.utils.data import DataLoader
from sklearn import metrics
from Bert_Attention import SharedBERTWithAttention


def train():
    model_name = 'best_graph_title2.pkl'
    log_dir = '/home/ypx/project/ypx/Agriculture_projects/分类实验/logs/graph_title2.log'
    test_path = '/home/ypx/project/ypx/Agriculture_projects/分类实验/datas/gra2/test_data.json'

    utils.set_logger(log_dir)
    model_path = r'/home/ypx/project/ypx/Agriculture_projects/models/bert-base-chinese'
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    learning_rate = 5e-6
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 获取到dataset
    label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    test_dataset = NewsDataset(test_path, tokenizer, label_map)
    logging.info('Loaded datasets.')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    bert_config = BertConfig.from_pretrained(model_path)
    num_labels = len(label_map)
    logging.info('testing model: {0}'.format(model_name))
    # 初始化模型
    model = SharedBERTWithAttention(bert_config, num_labels).to(device)
    model.load_state_dict(torch.load('../finetuning_models/{0}'.format(model_name), map_location=torch.device('cpu')))
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0

    model.eval()
    losses = 0  # 损失
    pred_labels = []
    true_labels = []
    valid_bar = tqdm(test_dataloader, ncols=100)
    with torch.no_grad():
        for batch in valid_bar:
            inputs, labels = batch
            title_input, entity_input = inputs
            title_input = {key: val.to(device) for key, val in title_input.items()}
            entity_input = {key: val.to(device) for key, val in entity_input.items()}
            # sentence_input = {key: val.to(device) for key, val in sentence_input.items()}
            labels = labels.to(device)
            output = model(title_input, entity_input)

            loss = criterion(output, labels.to(device))
            losses += loss.item()

            pred_label = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_label == labels).item() / len(pred_label)  # acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    average_loss = losses / len(test_dataloader)
    # print('\tLoss:', average_loss)
    logging.info('\tValid Loss: {0}'.format(average_loss))
    # 分类报告
    report = metrics.classification_report(true_labels, pred_labels, digits=5)
    # print('* Classification Report:')
    # print(report)
    logging.info('*{0} Valid Classification Report:'.format(model_name))
    logging.info('\n' + report)



if __name__ == '__main__':
    train()
