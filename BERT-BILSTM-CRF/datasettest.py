import os
import warnings

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import config
import utils
import logging

from dataLoader import NERDataset
from model import BertNER
from train import train

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

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    utils.set_logger(config.log_dir)
    device = config.device
    logging.info("device: {}".format(device))

    word_train, label_train = loadDataset(config.train_dir)
    for i in word_train:
        if len(i) == 223:
            print(i)
    word_dev, label_dev = loadDataset(config.valid_dir)