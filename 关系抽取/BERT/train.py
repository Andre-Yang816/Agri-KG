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
from model import BertClassifier


def train():
    # 特征融合
    #graph_multiAttention.pkl: title + entity + graph
    #best_model_attention: title + entity + sentence

    #best_entity_title.pkl:title + entity
    #best_graph_title: title + graph
    #best_graph_entity.pkl: graph + entity
    #需修改的三条
    model_name = 'RE_BERT.pkl'
    log_dir = '/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/logs/RE_BERT.log'
    train_path = '/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/data/train_data.json'
    val_path = '/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/data/val_data.json'


    utils.set_logger(log_dir)
    model_path = r'/home/ypx/project/ypx/Agriculture_projects/models/bert-base-chinese'
    batch_size = 8
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    bert_config = BertConfig.from_pretrained(model_path)
    num_labels = len(label_map)
    print(num_labels)
    logging.info('training model: {0}'.format(model_name))
    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0

    for epoch in range(1, epochs + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率

        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_bar = tqdm(train_dataloader, ncols=100)
        for inputs, label in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)
            # 传入数据，调用model.forward()
            input_ids = inputs['input_ids'].squeeze(1)
            attention_mask = inputs['attention_mask'].squeeze(1)
            token_type_ids = inputs['token_type_ids'].squeeze(1)
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            # 计算loss
            loss = criterion(output, label.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label

            acc = torch.sum(pred_labels == label.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        # print('\tTrain ACC: {0}'.format(average_acc), '\tTrain Loss: {0}'.format(average_loss))
        logging.info('Epoch %i train' % epoch)
        logging.info('\tTrain ACC:  {0}'.format(average_acc))
        logging.info('\tTrain Loss:  {0}'.format(average_loss))
        # 验证
        model.eval()
        losses = 0  # 损失
        pred_labels = []
        true_labels = []
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for inputs, label in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)
            input_ids = inputs['input_ids'].squeeze(1)
            attention_mask = inputs['attention_mask'].squeeze(1)
            token_type_ids = inputs['token_type_ids'].squeeze(1)
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            loss = criterion(output, label.to(device))
            losses += loss.item()

            pred_label = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_label == label.to(device)).item() / len(pred_label)  # acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(label.numpy().tolist())

        average_loss = losses / len(valid_dataloader)
        # print('\tLoss:', average_loss)
        logging.info('\tValid Loss: {0}'.format(average_loss))
        # 分类报告
        report = metrics.classification_report(true_labels, pred_labels, digits=5)
        # print('* Classification Report:')
        # print(report)
        logging.info('*{0} Valid Classification Report:'.format(model_name))
        logging.info('\n' + report)

        # f1 用来判断最优模型
        f1 = metrics.f1_score(true_labels, pred_labels, average='macro')


        # 判断并保存验证集上表现最好的模型
        if f1 > best_f1:
            logging.info('---Save the best model!---')
            best_f1 = f1
            torch.save(model.state_dict(), '../finetuning_models/{0}'.format(model_name))


if __name__ == '__main__':
    train()
