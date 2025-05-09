# coding: utf-8
# @File: train.py
# @Author: Peixin Yang
# @Description:
import logging
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import CNewsDataset
from tqdm import tqdm
from sklearn import metrics
import utils



def test():

    #务必修改模型名称
    current_train = 'entities'

    models = {
        'short': 'best_model_titles.pkl',
        'long': 'best_model_texts.pkl',
        'sentences': 'best_model_sentences.pkl',
        'entities': 'best_model_entities.pkl'
    }
    log_dirs = {
        'short': './logs/train_titles.log',
        'long': './logs/train_texts.log',
        'sentences': './logs/train_sentences.log',
        'entities': './logs/train_entities.log',
    }
    test_paths = {
        'short': 'shortText/test_titles.txt',
        'long': 'longText/test_texts.txt',
        'sentences': 'sentences/test_sentences.txt',
        'entities': 'entities/test_entities.txt',
    }
    model_name = models[current_train]
    log_dir = log_dirs[current_train]
    test_path = test_paths[current_train]
    utils.set_logger(log_dir)

    # 参数设置
    model_path = r'/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/model/bert-base-chinese/'
    data_path = r'/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/datasets/'
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(model_path)


    # 获取到dataset
    # test_dataset = CNewsDataset(data_path + 'test_long_texts_entities.txt', tokenizer)
    test_dataset = CNewsDataset(data_path + test_path, tokenizer)
    # 生成Batch
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(model_path)
    num_labels = len(test_dataset.labels)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)
    model.load_state_dict(torch.load('./finetuning_models/{0}'.format(model_name), map_location=torch.device('cpu')))
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 验证
    model.eval()
    losses = 0  # 损失
    pred_labels = []
    true_labels = []
    test_bar = tqdm(test_dataloader, ncols=100)
    for input_ids, token_type_ids, attention_mask, label_id in test_bar:
        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            token_type_ids=token_type_ids.to(device),
        )

        loss = criterion(output, label_id.to(device))
        losses += loss.item()

        pred_label = torch.argmax(output, dim=1)  # 预测出的label
        acc = torch.sum(pred_label == label_id.to(device)).item() / len(pred_label)  # acc
        test_bar.set_postfix(loss=loss.item(), acc=acc)

        pred_labels.extend(pred_label.cpu().numpy().tolist())
        true_labels.extend(label_id.numpy().tolist())

    average_loss = losses / len(test_dataloader)
    print('\tLoss:', average_loss)
    # 分类报告
    report = metrics.classification_report(true_labels, pred_labels, labels=test_dataset.labels_id,
                                               target_names=test_dataset.labels, digits=5)

    logging.info('*{0} Test Classification Report:'.format(model_name))
    logging.info('\n'+ report)


if __name__ == '__main__':
    test()