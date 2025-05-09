import json
import pandas as pd
from glob import glob


# 读取两个 CSV 文件
df1 = pd.read_csv("new_matched_triples.csv", encoding="utf-8-sig")
df2 = pd.read_csv("unique_relations.csv", encoding="utf-8-sig")

# 合并并去重
df_combined = pd.concat([df1, df2]).drop_duplicates()

# 保存合并后的文件
df_combined.to_csv("final_relations.csv", index=False, encoding="utf-8-sig")

print("CSV 文件已合并: merged_file.csv")