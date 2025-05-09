valid_relations = {"属于","病害","虫害","治疗","种","饲料","目","属","纲"}
import networkx as nx
import pandas as pd

# 1️⃣ 读取 CSV 数据
file_path = "final_triples.csv"
df = pd.read_csv(file_path)

# 2️⃣ 去除 entity1 或 entity2 为 "生物" 的行
df = df[(df["entity1"] != "生物") & (df["entity2"] != "生物")]

# 2️⃣ 构建有向图
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["entity1"], row["entity2"], relation=row["relation"])

# 3️⃣ 进行 PRA 路径搜索（找到两跳以内的可能关系）
new_triples = []
for node in G.nodes():
    for neighbor in G.neighbors(node):
        for next_neighbor in G.neighbors(neighbor):
            # 如果 (node, next_neighbor) 没有直接连接，则补全关系
            if not G.has_edge(node, next_neighbor):
                # relation = G[node][neighbor]["relation"] + " → " + G[neighbor][next_neighbor]["relation"]
                relation = G[neighbor][next_neighbor]["relation"]
                if relation in valid_relations:
                    if node != next_neighbor:
                        new_triples.append((node, relation, next_neighbor))

# 4️⃣ 转换为 DataFrame
new_df = pd.DataFrame(new_triples, columns=["entity1", "relation", "entity2"])

# 5️⃣ 合并原始知识图谱，去重后保存
final_df = pd.concat([df, new_df]).drop_duplicates()
final_df.to_csv("completed_triples.csv", index=False)

print(f"�� 补全了 {len(new_triples)} 条新关系，并已保存至 completed_triples.csv ✅")
