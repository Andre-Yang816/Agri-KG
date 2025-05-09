import pandas as pd
from Levenshtein import distance as levenshtein_distance
from difflib import get_close_matches

# 读取预测实体集 E1（每行一个实体）
with open("sorted_entities.txt", "r", encoding="utf-8") as f:
    predicted_entities = [line.strip() for line in f.readlines()]

# 读取知识图谱实体集 E2（每行一个实体）
with open("/home/ypx/project/ypx/Agriculture_projects/知识图谱/知识图谱/code/entities.txt", "r", encoding="utf-8") as f:
    knowledge_base = [line.strip() for line in f.readlines()]

# 设定匹配阈值（0.7 表示相似度至少达到 70%）
SIMILARITY_THRESHOLD = 0.7

def get_best_match(entity, knowledge_base, threshold=SIMILARITY_THRESHOLD):
    """ 计算 entity 在知识图谱中的最佳匹配实体 """
    matches = get_close_matches(entity, knowledge_base, n=1, cutoff=threshold)
    return matches[0] if matches else None

# 进行实体链接
linked_entities = {e: get_best_match(e, knowledge_base) for e in predicted_entities}
output_file = "linked_entities.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for entity, linked in linked_entities.items():
        if linked:
            f.write(linked+'\n')
# # 保存实体链接结果
# output_file = "linked_entities.csv"
# df = pd.DataFrame(linked_entities.items(), columns=["Predicted_Entity", "Linked_Entity"])
# df.to_csv(output_file, index=False, encoding="utf-8")
#
# print(f"实体链接完成，结果已保存至 {output_file}")
