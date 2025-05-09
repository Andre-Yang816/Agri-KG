import os
import warnings
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, AutoModelForTokenClassification

import config
import utils
import logging
from dataLoader import NERDataset
from model import ERNIE_BiLSTM_CRF
from train import train, evaluate
warnings.filterwarnings('ignore')

def loadDataset(f_path):
    MAX_LEN = 512 - 2
    with open(f_path, 'r', encoding='utf-8') as f:
        lines = [line.split('\n')[0] for line in f.readlines() if len(line.strip()) != 0]

    tags = [line.split('\t')[1] for line in lines]
    words = [line.split('\t')[0] for line in lines]
    sents = []
    labels = []
    word, tag = [], []
    for char, t in zip(words, tags):
        if char != '。':
            word.append(char)
            tag.append(t)
        else:
            if len(word) < MAX_LEN:
                sents.append(word)
                labels.append(tag)
            else:
                sents.append(word[:MAX_LEN])
                labels.append(tag[:MAX_LEN])
            word, tag = [], []
    return sents, labels

def compute_label_weights(all_labels, label2id):
    all_label_ids = [label2id[label] for sentences in all_labels for label in sentences]
    total_count = len(all_label_ids)
    label_counts = Counter(all_label_ids)
    weights = [total_count/label_counts.get(i, 1) for i in range(len(label2id))]
    weights = torch.tensor(weights)
    weights = weights / weights.sum() * len(weights)
    return weights



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    utils.set_logger(config.log_dir)
    device = config.device
    logging.info("device: {}".format(device))

    word_train, label_train = loadDataset(config.train_dir)
    word_dev, label_dev = loadDataset(config.valid_dir)

    # build dataset
    train_dataset = NERDataset(word_train, label_train, config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")
    # Prepare model
    label_weights = compute_label_weights(label_train , config.label2id)
    config.class_weights = label_weights
    model = ERNIE_BiLSTM_CRF(config)


    bert_optimizer = list(model.ernie.named_parameters())
    lstm_optimizer = list(model.bilstm.named_parameters())
    classifier_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
        {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, config.learning_rate)


    train_steps_per_epoch = train_size // config.batch_size

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)