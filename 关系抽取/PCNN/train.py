import logging
import os

from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np

import utils
from dataset import NewsDataset
from PCNN import PCNN


def train(model, dataloader, optimizer, device):
    model = model.to(device)
    model.train()

    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch
        input_ids = inputs['input_ids']
        optimizer.zero_grad()
        outputs = model(input_ids.to(device))
        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Average training loss: {avg_loss:.4f}")



def evaluate(model, dataloader, device, model_name):
    global best_f1  # 允许修改全局变量
    model = model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            input_ids = inputs['input_ids'].squeeze(1)

            outputs = model(input_ids.to(device))
            _, preds = torch.max(outputs, dim=1)  # 获取预测类别
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算分类报告
    report = classification_report(true_labels, predictions, digits=4)

    # 计算宏平均 F1 分数
    f1 = metrics.f1_score(true_labels, predictions, average='macro')

    print(f"Validation Macro F1-score: {f1:.4f}")
    print("Classification Report:\n",
          classification_report(true_labels, predictions, digits=4))

    # **如果当前 F1 分数是最佳的，则保存模型**
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), '../finetuning_models/{0}'.format(model_name))
        print(f"New best model saved with F1: {best_f1:.4f}")

    return f1, report


if __name__ == '__main__':
    # 特征融合
    #graph_multiAttention.pkl: title + entity + graph
    #best_model_attention: title + entity + sentence

    #best_entity_title.pkl:title + entity
    #best_graph_title: title + graph
    #best_graph_entity.pkl: graph + entity
    #需修改的三条
    model_name = 'RE_PCNN.pkl'
    log_dir = '/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/logs/PCNN.log'

    train_path = '/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/data/train_data.json'
    val_path = '/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/data/val_data.json'
    utils.set_logger(log_dir)
    model_path = r'/home/ypx/project/ypx/Agriculture_projects/models/bert-base-chinese'
    batch_size = 8
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    epochs = 5
    learning_rate = 5e-6
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 获取到dataset
    label_map = {"属于": 0, "防治": 1, "感染": 2, "别名": 3, "被危害": 4, "科":5, "导致":6, "病原":7, "目":8, "纲":9, "不良反应":10}
    # "属于": 0, "防治": 1, "感染": 2, "别名": 3, "被危害": 4, "科":5, "导致":6, "病原":7, "目":8, "纲":9, "不良反应":10


    train_dataset = NewsDataset(train_path, tokenizer, label_map)
    valid_dataset = NewsDataset(val_path, tokenizer, label_map)
    logging.info('Loaded datasets.')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    num_labels = len(label_map)
    print(num_labels)
    logging.info('training model: {0}'.format(model_name))

    # 设置超参数
    MAX_LEN = 512
    BATCH_SIZE = 16
    EPOCHS = 5
    HIDDEN_DIM = 512  # 可以选择合适的维度（例如，BERT的输出维度）
    KERNEL_SIZE = 1  # 可以根据需要调整
    NUM_LABELS = 11  # 假设你有3个关系类别
    DROPOUT = 0.3
    # 初始化PCNN模型
    model = PCNN(hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, num_labels=NUM_LABELS, dropout=DROPOUT)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # 初始化最佳 F1 分数
    best_f1 = 0.0
    # 训练模型
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train(model, train_dataloader, optimizer, device)

        # 验证
        print("Evaluating...")
        evaluate(model, valid_dataloader, device, model_name)

