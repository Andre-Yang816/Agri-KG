# coding: utf-8
# @File: train.py
# @Author: Peixin Yang
# @Description:
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import CNewsDataset
from tqdm import tqdm
from sklearn import metrics
import logging
import utils


def main():
    #务必修改这里
    current_train = 'long'

    models = {
        'short':'best_model_titles.pkl',
        'long':'best_model_texts.pkl',
        'sentences':'best_model_sentences.pkl',
        'entities': 'best_model_entities.pkl'
    }
    log_dirs = {
        'short': './logs/train_titles.log',
        'long': './logs/train_texts.log',
        'sentences': './logs/train_sentences.log',
        'entities':'./logs/train_entities.log',
    }
    train_paths = {
        'short': 'shortText/train_titles.txt',
        'long': 'longText/train_texts.txt',
        'sentences': 'sentences/train_sentences.txt',
        'entities': 'entities/train_entities.txt',
    }
    val_paths = {
        'short': 'shortText/val_titles.txt',
        'long': 'longText/val_texts.txt',
        'sentences': 'sentences/val_sentences.txt',
        'entities': 'entities/val_entities.txt',
    }
    model_name = models[current_train]
    log_dir = log_dirs[current_train]
    train_path = train_paths[current_train]
    val_path = val_paths[current_train]
    utils.set_logger(log_dir)
    logging.info('loading model: {0}'.format(model_name))
    # 参数设置
    model_path = r'/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/model/bert-base-chinese/'
    data_path = r'/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/datasets/'
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 20
    learning_rate = 5e-6
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 获取到dataset
    train_dataset = CNewsDataset(data_path + train_path, tokenizer)
    valid_dataset = CNewsDataset(data_path + val_path, tokenizer)
    # test_dataset = CNewsDataset(data_path + 'test_shortText.txt', tokenizer)


    # 生成Batch
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件

    bert_config = BertConfig.from_pretrained(model_path)
    num_labels = len(train_dataset.labels)
    logging.info('training model: {0}'.format(model_name))
    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(1, epochs+1):
        losses = 0      # 损失
        accuracy = 0    # 准确率

        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)
            # 传入数据，调用model.forward()
            output = model(
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device), 
                token_type_ids=token_type_ids.to(device), 
            )

            # 计算loss
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)   # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels) #acc
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
        losses = 0      # 损失
        pred_labels = []
        true_labels = []
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id  in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device), 
                token_type_ids=token_type_ids.to(device), 
            )
            
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_label = torch.argmax(output, dim=1)   # 预测出的label
            acc = torch.sum(pred_label == label_id.to(device)).item() / len(pred_label) #acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(label_id.numpy().tolist())

        average_loss = losses / len(valid_dataloader)
        # print('\tLoss:', average_loss)
        logging.info('\tValid Loss: {0}'.format(average_loss))
        # 分类报告
        report = metrics.classification_report(true_labels, pred_labels, labels=valid_dataset.labels_id, target_names=valid_dataset.labels, digits=5)
        # print('* Classification Report:')
        # print(report)
        logging.info('*{0} Valid Classification Report:'.format(model_name))
        logging.info('\n'+report)

        # f1 用来判断最优模型
        f1 = metrics.f1_score(true_labels, pred_labels, labels=valid_dataset.labels_id, average='micro')
        
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # 判断并保存验证集上表现最好的模型
        if f1 > best_f1:
            logging.info('---Save the best model!---')
            best_f1 = f1
            torch.save(model.state_dict(), './finetuning_models/{0}'.format(model_name))
        
if __name__ == '__main__':
    main()