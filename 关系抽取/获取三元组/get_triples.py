def load_entities(entity_file):
    """加载entities.txt中的实体"""
    entities = set()
    with open(entity_file, "r", encoding="utf-8") as f:
        for line in f:
            entity = line.strip()
            if entity:
                entities.add(entity)
    return entities

def load_baike_triples(baike_file):
    """加载baike_triples.txt中的三元组"""
    triples = []
    with open(baike_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")  # 假设三元组是以 tab 分隔
            if len(parts) == 3:
                triples.append(parts)  # [实体1, 关系, 实体2]
    return triples

def match_entities_with_triples(entities, triples, output_file):
    """直接匹配entities.txt中的实体与baike_triples.txt中的三元组"""
    with open(output_file, "w", encoding="utf-8") as f_out:
        for triple in triples:
            entity1, relation, entity2 = triple
            # 检查entity1和entity2是否与实体集中的任何实体匹配
            if entity1 in entities and entity2 in entities:
                print(triple)
                if entity1 != entity2:
                    f_out.write(",".join(triple) + "\n")  # 保留该三元组

# 文件路径
entities_path = "sorted_entities.txt"
baike_path = "/home/ypx/project/ypx/Agriculture_projects/知识图谱/知识图谱/baike_triples.txt"
output_path = "matched_triples309.txt"

# 读取数据
entities = load_entities(entities_path)
baike_triples = load_baike_triples(baike_path)

# 匹配实体和三元组
match_entities_with_triples(entities, baike_triples, output_path)

print("matched_triples.txt 生成完毕！")
