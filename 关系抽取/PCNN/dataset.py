import json

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer



def preprocess(filename, tokenizer, max_length=512):
    inputs = []
    text_inputs = []
    entity_inputs = []
    sentences_inputs = []
    labels = []
    print('loading data from:', filename)
    with open(filename, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    for item in tqdm(datas, ncols=100):
        entity1 = item["实体1"]
        entity2 = item["实体2"]
        sentence = item["句子"]
        label = item["关系"][0]
        # 使用 [SEP] 进行拼接
        text = f"{entity1} [SEP] {entity2} [SEP] {sentence}"
        # 转换为 BERT 需要的格式
        encoded_text = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        # inputs.append(encoded)
        text_inputs.append(encoded_text)
        labels.append(label)

    return text_inputs, labels

class NewsDataset(Dataset):
    def __init__(self, filename, tokenizer, label_map, max_length=512):
        self.inputs, self.labels = preprocess(filename, tokenizer, max_length)
        self.label_map = label_map

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        text_input = self.inputs[idx]
        label = self.label_map[self.labels[idx]]

        return text_input, torch.tensor(label)



