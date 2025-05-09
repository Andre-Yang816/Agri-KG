import json

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
import config



def preprocess(filename, tokenizer, max_length=config.MAX_LEN):
    inputs = []
    title_inputs = []
    entity_inputs = []
    sentences_inputs = []
    gra_inputs = []
    labels = []
    print('loading data from:', filename)
    with open(filename, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    for item in tqdm(datas, ncols=100):
        title = item["Title"]
        entity = item["Entity"]
        gra = item["gra"]
        sentence = item["Sentence"]
        label = item["Label"]
        # 使用 [SEP] 进行拼接
        # text = f"{title} [SEP] {entity} [SEP] {sentence}"
        # 转换为 BERT 需要的格式
        encoded_title = tokenizer(title, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        encoded_entity = tokenizer(entity, padding='max_length', truncation=True, max_length=max_length,
                                  return_tensors="pt")
        encoded_sentence = tokenizer(sentence, padding='max_length', truncation=True, max_length=max_length,
                                  return_tensors="pt")
        encoded_gra = tokenizer(gra, padding='max_length', truncation=True, max_length=max_length,
                                     return_tensors="pt")
        # inputs.append(encoded)
        title_inputs.append(encoded_title)
        entity_inputs.append(encoded_entity)
        sentences_inputs.append(encoded_sentence)
        gra_inputs.append(encoded_gra)
        labels.append(label)

    return (title_inputs, entity_inputs, sentences_inputs, gra_inputs), labels

class NewsDataset(Dataset):
    def __init__(self, filename, tokenizer, label_map, max_length=config.MAX_LEN):
        self.inputs, self.labels = preprocess(filename, tokenizer, max_length)
        self.label_map = label_map

    def __len__(self):
        return len(self.inputs[0])

    def __getitem__(self, idx):
        title_input = self.inputs[0][idx]
        entity_input = self.inputs[1][idx]
        sentence_input = self.inputs[2][idx]
        gra_input = self.inputs[3][idx]
        label = self.label_map[self.labels[idx]]

        return (title_input, entity_input, sentence_input, gra_input), torch.tensor(label)



