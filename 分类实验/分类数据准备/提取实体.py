# 读取关系图谱文件，提取所有唯一实体
def extract_entities(graph_file, entity_output_file):
    entities = set()
    with open(graph_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')  # 假设是以tab分隔
            if len(parts) == 3:
                entities.add(parts[0])  # 第一列实体
                entities.add(parts[2])  # 第三列实体

    with open(entity_output_file, 'w', encoding='utf-8') as f:
        for entity in sorted(entities):
            f.write(entity + '\n')

graph_file = "matched_triples.csv"
entity_output_file = "unique_entities.txt"
extract_entities(graph_file, entity_output_file)
