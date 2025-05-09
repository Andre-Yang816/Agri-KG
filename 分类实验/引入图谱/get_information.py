import json
import re

# 清洗函数：只保留中文字符并限制最大长度
def clean_and_limit_text(text, max_length=1024):
    cleaned_text = ','.join(re.findall(r'[\u4e00-\u9fff\u3001-\u303F\uFF00-\uFFEF0-9]', text))  # 保留中文字符
    return cleaned_text[:max_length]  # 限制字符长度

# 扩展知识图谱
def expand_knowledge_graph(data, knowledge_graph, max_length=1024):
    # 读取知识图谱并构建关系字典
    relation_dict = {}
    for line in knowledge_graph:
        entity1, relation, entity2 = line.strip().split('\t')
        if entity1 not in relation_dict:
            relation_dict[entity1] = []
        relation_dict[entity1].append( f'{entity2}')
        if entity2 not in relation_dict:
            relation_dict[entity2] = []
        relation_dict[entity2].append(f'{entity1}')

    # 处理JSON数据，扩展图谱信息
    for item in data:
        entities = item["Entity"].split(', ')
        graph_info = []
        for entity in entities:
            if entity in relation_dict:
                graph_info.extend(relation_dict[entity])
        item["knowledge_graph"] = ",".join(graph_info) if graph_info else ""
        # 在扩展知识图谱后直接清洗 "gra" 字段并限制长度
        item["knowledge_graph"] = clean_and_limit_text(item["knowledge_graph"], max_length)

    return data


# 读取JSON数据
with open("/home/ypx/project/ypx/Agriculture_projects/分类实验/引入图谱/processed_news.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 读取通用知识图谱数据
with open("/home/ypx/project/ypx/Agriculture_projects/知识图谱/知识图谱/baike_triples.txt", "r", encoding="utf-8") as f:
    knowledge_graph_data = f.readlines()

# 处理数据
new_json_data = expand_knowledge_graph(json_data, knowledge_graph_data, max_length=1024)

# 只保留需要的字段
filtered_data = [
    {"Title": item["Title"], "Entity": item["Entity"], "Label": item["Label"], "gra": item["knowledge_graph"]} for item in
    new_json_data]

# 保存为新的JSON文件
with open("GeneraKG.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print("扩展完成，已保存为 GeneraKG.json")
