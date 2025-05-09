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
    model_name = 'best_graph_title_GKG.pkl'
    log_dir = '/home/ypx/project/ypx/Agriculture_projects/分类实验/logs/graph_title_GKG.log'
    train_path = '/home/ypx/project/ypx/Agriculture_projects/分类实验/datas/Generate_KG/train_data.json'
    val_path = '/home/ypx/project/ypx/Agriculture_projects/分类实验/datas/Generate_KG/val_data.json'


    utils.set_logger(log_dir)
    model_path = r'/home/ypx/project/ypx/Agriculture_projects/models/bert-base-chinese'
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    epochs = 10
    learning_rate = 5e-6
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 获取到dataset
    label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    train_dataset = NewsDataset(train_path, tokenizer, label_map)
    valid_dataset = NewsDataset(val_path, tokenizer, label_map)
    logging.info('Loaded datasets.')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    bert_config = BertConfig.from_pretrained(model_path)
    num_labels = len(label_map)
    print(num_labels)
    logging.info('training model: {0}'.format(model_name))
    # 初始化模型
    model = SharedBERTWithAttention(bert_config, num_labels).to(device)

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

        accumulation_steps = 4
        optimizer.zero_grad()
        for batch in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)
            # 传入数据，调用model.forward()
            inputs, labels = batch
            title_input, entity_input = inputs
            title_input = {key: val.to(device) for key, val in title_input.items()}
            entity_input = {key: val.to(device) for key, val in entity_input.items()}
            # sentence_input = {key: val.to(device) for key, val in sentence_input.items()}
            labels = labels.to(device)
            output = model(title_input, entity_input)

            # 计算loss
            loss = criterion(output, labels)
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            # labels = torch.tensor(labels)
            acc = torch.sum(pred_labels == labels).item() / len(pred_labels)  # acc
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
        with torch.no_grad():
            for batch in valid_bar:
                valid_bar.set_description('Epoch %i valid' % epoch)
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
        f1 = metrics.f1_score(true_labels, pred_labels, average='micro')


        # 判断并保存验证集上表现最好的模型
        if f1 > best_f1:
            logging.info('---Save the best model!---')
            best_f1 = f1
            torch.save(model.state_dict(), '../finetuning_models/{0}'.format(model_name))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
