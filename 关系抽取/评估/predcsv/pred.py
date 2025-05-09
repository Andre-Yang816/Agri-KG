import json
import pandas as pd
from glob import glob

# 读取所有 JSON 文件
json_files = glob("../*.json")
unique_relations = set()

for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            entity1 = entry["实体1"]
            entity2 = entry["实体2"]
            relation = entry["关系"][0]  # 取关系的第一个元素
            unique_relations.add((entity1, entity2, relation))

# 转换为 DataFrame 并保存为 CSV
df = pd.DataFrame(unique_relations, columns=["实体1", "关系", "实体2"])
df.to_csv("unique_relations.csv", index=False, encoding="utf-8-sig")

print("CSV 文件已生成: unique_relations.csv")
