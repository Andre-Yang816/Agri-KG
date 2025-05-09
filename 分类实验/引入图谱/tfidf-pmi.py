import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import jieba  # 中文分词工具
import re

# 加载 long_texts.txt 文件
long_texts = []
labels = []
with open('/home/ypx/project/ypx/Agriculture_projects/分类实验/分类数据准备/new_long_texts.txt', 'r', encoding='utf-8') as f:
    for line in f:
        label, text = line.strip().split("\t", 1)
        labels.append(int(label))  # 将标签转为整数
        long_texts.append(text)

# 加载 JSON 数据
with open('processed_news.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# with open('stop.txt', 'r', encoding='utf-8') as f:
#     stop = f.readlines()
# 确保文本数据与标签匹配
assert len(data) == len(long_texts), "数据长度不匹配"


# 中文文本分词
def chinese_tokenizer(text):
    return list(jieba.cut(text))

# 使用 TF-IDF 计算每个文本中的重要词汇
vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
X = vectorizer.fit_transform(long_texts)

# 获取所有特征词（即词汇表）
features = vectorizer.get_feature_names_out()

# 创建一个字典，用于存储每个词汇的TF-IDF值
tfidf_dict = {word: X[:, i].sum() for i, word in enumerate(features)}

# 打印一下计算出的TF-IDF值的前几个词（用于检查）
print(sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:50])
def compute_pmi(entity, text, total_count, entity_count):
    """
    计算PMI值
    """
    # 计算联合概率 P(w, c)
    co_occur = text.count(entity)  # 实体和文本共现的次数
    pmi = np.log((co_occur + 1) / (entity_count + 1) )
    return pmi


# 假设我们已经获得了所有实体在文本中的统计信息
entity_pmi_scores = {}

for item, text in zip(data, long_texts):
    entity_list = item["Entity"].split(", ")

    for entity in entity_list:
        pmi_score = compute_pmi(entity, text, len(long_texts), len(entity_list))
        entity_pmi_scores[entity] = pmi_score

# 打印出所有实体的PMI值
print(entity_pmi_scores)

# 结合 PMI 和 TF-IDF 进行筛选
combined_scores = {}

for entity, pmi_score in entity_pmi_scores.items():
    # 获取该实体的TF-IDF值
    tfidf_score = tfidf_dict.get(entity, 0)

    # 结合 TF-IDF 和 PMI 值（简单加权）
    combined_scores[entity] = pmi_score * tfidf_score

# 打印最重要的实体（按加权评分排序）
filtered_entities = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
print(filtered_entities[:10])


# 中文文本分句
def chinese_split_sentences(text):
    return re.split('。|！|？|；', text)


# 添加筛选后的实体和实体所在句子到 JSON 数据中
for item, text in zip(data, long_texts):
    entities_in_text = item["Entity"].split(", ")
    filtered_entities_in_text = []
    sentences = chinese_split_sentences(text)  # 中文分句

    for entity in entities_in_text:
        if entity in combined_scores and combined_scores[entity] > 0:
            # 筛选出该实体所在的句子
            for sentence in sentences:
                if entity in sentence:
                    filtered_entities_in_text.append((entity, sentence))

    item["filted_entities"] = [entity for entity, _ in filtered_entities_in_text]
    item["entity_sentences"] = [sentence for _, sentence in filtered_entities_in_text]

# 保存新的 JSON 数据
with open('enhanced_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
