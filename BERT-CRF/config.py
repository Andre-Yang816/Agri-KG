import os
import torch

data_dir = os.getcwd() + '/dataset/'
train_dir = data_dir + 'all_datas/train.txt'
valid_dir = data_dir + 'all_datas/valid.txt'
test_dir = data_dir + 'all_datas/test.txt'
files = ['train', 'test']

bert_model = './model/bert-base-chinese/'

#roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/experiments/case/bad_case.txt'

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 16
epoch_num = 30
min_epoch_num = 5
patience = 0.0002
patience_num = 10

#bilstm
lstm_embedding_size = 768
hidden_size = 512
lstm_dropout_prob = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

labels = ['Crop', 'PestiCide', 'CropDisease', 'CropPest', 'Animal', 'AnimalDisease', 'AnimalPest', 'Drugs']

label2id = {
    'O': 0,
    'B-Crop':1,
    'I-Crop':2,  # Crop
    'B-PestiCide':3,
    'I-PestiCide':4,  # PestiCide
    'B-CropDisease':5,
    'I-CropDisease':6,  # Disease
    'B-CropPest':7,
    'I-CropPest':8,  # pest
    'B-Animal':9,
    'I-Animal':10,
    'B-AnimalDisease':11,
    'I-AnimalDisease':12,
    'B-AnimalPest':13,
    'I-AnimalPest':14,
    'B-Drugs':15,
    'I-Drugs':16,
}
num_labels = len(label2id)

id2label = {_id: _label for _label, _id in list(label2id.items())}
