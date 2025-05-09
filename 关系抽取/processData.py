import re
import torch
from transformers import BertTokenizer
from kg_classify_ERNIE.ernie_model import ERNIE_BiLSTM_CRF
from utils import get_input_token_starts, get_entities
import config



tokenizer = BertTokenizer.from_pretrained(config.vocab)
ner_model = ERNIE_BiLSTM_CRF(config)
weights = torch.load(config.model_dir)
ner_model.load_state_dict(weights)
ner_model.eval()

# 假设你有一个实体识别模型的接口
def ner_predict(model, sentence):
    """
    使用命名实体识别模型对句子进行实体预测
    返回: [(实体, 类型), ...]
    """
    # 假设模型输出实体及其类型
    # entities = model.predict(sentence)
    token_starts = get_input_token_starts(sentence, tokenizer)
    tokens = tokenizer.encode(sentence, truncation=True, add_special_tokens=False, max_length=510)
    test_input = torch.tensor([tokens], dtype=torch.long)
    token_starts = torch.tensor([token_starts], dtype=torch.long)
    test_input = (test_input, token_starts)
    tag_scores = model.forward(test_input)
    labels_pred = model.crf.decode(tag_scores[0])
    pred_labels = [[config.id2label.get(idx) for idx in indices] for indices in labels_pred]
    pred_entities = get_entities(pred_labels)

    return pred_entities


def load_entities(entities_file):
    """
    从文件加载实体集合
    返回: 一个实体集合（去重）
    """
    with open(entities_file, 'r') as f:
        entities = set(line.strip() for line in f.readlines())
    return entities


def format_sentence(sentence, entity1, entity2):
    """
    格式化句子，添加实体标记
    """
    formatted = sentence.replace(entity1, f"[E1]{entity1}[/E1]")
    formatted = formatted.replace(entity2, f"[E2]{entity2}[/E2]")
    return formatted


def process_sentences(sentence_list, entities_file, output_file):
    """
    处理句子列表，识别实体，筛选符合条件的实体对，写入到输出文件
    """
    # 加载实体集合
    valid_entities = load_entities(entities_file)

    with open(output_file, 'w') as f_out:
        for sentence in sentence_list:
            # 第一步：实体识别
            entities = ner_predict(ner_model, sentence)

            # 第二步：筛选实体，保留在 valid_entities 中出现的实体
            filtered_entities = [entity for entity, _ in entities if entity in valid_entities]

            # 第三步：如果句子有多个实体，构建实体对并写入文件
            if len(filtered_entities) > 1:
                # 生成所有实体对
                for i in range(len(filtered_entities)):
                    for j in range(i + 1, len(filtered_entities)):
                        entity1, entity2 = filtered_entities[i], filtered_entities[j]
                        formatted_sentence = format_sentence(sentence, entity1, entity2)
                        f_out.write(f"{formatted_sentence},{entity1},{entity2}\n")
            # 如果句子只有一个或没有符合条件的实体，跳过该句子
            elif len(filtered_entities) == 0:
                continue


# 示例数据
sentence_list = [
    '养蟹池塘如何种植苦草,4月中旬水温回升至16以上时,播种每亩播种苦草籽50克。',
    '如果种姜选择不当，会导致生姜出苗不齐，长势较弱，严重影响产量的提高。'
]

# 运行处理函数
process_sentences(sentence_list, './数据准备/实体集/entities.txt', './数据准备/待标注集/output.txt')
