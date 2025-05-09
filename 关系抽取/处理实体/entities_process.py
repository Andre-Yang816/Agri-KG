import re
from difflib import SequenceMatcher

def load_entities(entity_file):
    """加载entities.txt中的实体"""
    entities = set()
    with open(entity_file, "r", encoding="utf-8") as f:
        for line in f:
            entity = line.strip()
            if entity:
                entities.add(entity)
    return entities

def load_baike_entities(baike_file):
    """加载baike_triples.txt中的头实体和尾实体"""
    baike_entities = set()
    with open(baike_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")  # 假设三元组是以 tab 分隔
            if len(parts) == 3:
                baike_entities.add(parts[0])  # 头实体
                baike_entities.add(parts[2])  # 尾实体
    return baike_entities

def similar(a, b):
    """计算两个字符串的相似度（基于编辑距离）"""
    return SequenceMatcher(None, a, b).ratio()

def filter_entities(entities, baike_entities, similarity_threshold=0.8):
    """筛选出在百科中出现过的实体，或者相似度较高的实体"""
    filtered_entities = set()

    for entity in entities:
        if entity in baike_entities:
            # 直接匹配
            filtered_entities.add(entity)
        # else:
        #     # 计算相似度
        #     for baike_entity in baike_entities:
        #         if similar(entity, baike_entity) >= similarity_threshold:
        #             filtered_entities.add(entity)
        #             break  # 一旦找到相似的就停止

    return filtered_entities

def save_filtered_entities(filtered_entities, output_file):
    """保存筛选后的实体"""
    with open(output_file, "a", encoding="utf-8") as f:
        for entity in filtered_entities:
            f.write(entity + "\n")

# 文件路径
entities_path = "entities.txt"
baike_path = "/home/ypx/project/ypx/Agriculture_projects/知识图谱/知识图谱/baike_triples.txt"
output_path = "filtered_entities.txt"

# 读取数据
entities = load_entities(entities_path)
baike_entities = load_baike_entities(baike_path)

# 筛选实体
filtered_entities = filter_entities(entities, baike_entities, similarity_threshold=0.8)

# 保存结果
save_filtered_entities(filtered_entities, output_path)

print("filtered_entities.txt 生成完毕！")
