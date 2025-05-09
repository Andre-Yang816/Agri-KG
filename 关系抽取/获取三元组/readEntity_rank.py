def load_entities(entity_file):
    """加载entities.txt中的实体"""
    entities = set()
    with open(entity_file, "r", encoding="utf-8") as f:
        for line in f:
            entity = line.strip()
            if entity:
                entities.add(entity)
    rank_entitiy = sorted(entities, key=len)
    with open("sorted_entities.txt", "w", encoding="utf-8") as f:
        for word in rank_entitiy:
            f.write(word + "\n")

# 文件路径
entities_path = "filtered_entities.txt"
entities = load_entities(entities_path)