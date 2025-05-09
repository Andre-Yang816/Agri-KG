import networkx as nx
import pandas as pd
from collections import defaultdict

# 1️⃣ 读取 CSV 数据
file_path = "final_triples.csv"
df = pd.read_csv(file_path)

# 2️⃣ 去除 entity1 或 entity2 为 "生物" 的行
df = df[(df["entity1"] != "生物") & (df["entity2"] != "生物")]

# 3️⃣ 构建有向知识图谱
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["entity1"], row["entity2"], relation=row["relation"])

# 4️⃣ 计算 **实体的 PageRank**
node_pagerank = nx.pagerank(G)

# 5️⃣ 计算 **关系的 PageRank**（关系重要性）
relation_scores = defaultdict(float)
for entity1, entity2, data in G.edges(data=True):
    relation = data["relation"]
    # 计算关系重要性 = 连接的两个实体 PageRank 平均值
    relation_scores[relation] += (node_pagerank.get(entity1, 0) + node_pagerank.get(entity2, 0)) / 2

# 归一化
max_score = max(relation_scores.values(), default=1)
for rel in relation_scores:
    relation_scores[rel] /= max_score  # 归一化到 [0,1]

# 6️⃣ 执行 PRA 并结合关系重要性确定最终关系
new_triples = []
for node in G.nodes():
    for neighbor in G.neighbors(node):
        for next_neighbor in G.neighbors(neighbor):
            if not G.has_edge(node, next_neighbor):  # 只有当 (node, next_neighbor) 没有直接连接时，才进行补全
                relation1 = G[node][neighbor]["relation"]
                relation2 = G[neighbor][next_neighbor]["relation"]

                # 计算最终关系权重 = 关系1的PageRank * 0.5 + 关系2的PageRank
                weight1 = relation_scores.get(relation1, 0)
                weight2 = relation_scores.get(relation2, 0)
                final_score1 = weight1 * 0.5 + weight2
                final_score2 = weight2 * 0.5 + weight1

                # 选择得分最高的关系
                final_relation = relation1 if final_score1 >= final_score2 else relation2

                new_triples.append((node, final_relation, next_neighbor))

# 7️⃣ 转换为 DataFrame 并保存
new_df = pd.DataFrame(new_triples, columns=["entity1", "relation", "entity2"])
final_df = pd.concat([df, new_df]).drop_duplicates()
final_df.to_csv("completed_triples_relation_pagerank.csv", index=False)

print(f"✅ 补全了 {len(new_triples)} 条新关系，并已保存至 completed_triples_relation_pagerank.csv")
