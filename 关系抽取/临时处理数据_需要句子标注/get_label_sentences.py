import re

def load_entities(entity_file):
    """加载实体列表并按长度排序，去除空白字符"""
    entities = set()
    with open(entity_file, "r", encoding="utf-8") as f:
        for line in f:
            entity = line.strip()
            if entity:
                entities.add(entity)
    return sorted(entities, key=len, reverse=True)  # 按长度排序，优先匹配长实体

def annotate_sentences(input_file, entities, output_file):
    """对句子进行实体匹配并标注，同时添加实体列表和实体数量"""
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            sentence = line.strip()
            found_entities = set()  # 记录该句子中的实体

            # 遍历实体，找到后替换加标注（避免重复标注）
            for entity in entities:
                if entity in sentence:
                    found_entities.add(entity)
                    # 使用正则确保实体 **未被标注过** 再进行标注
                    sentence = re.sub(rf'(?<!\[E\]){re.escape(entity)}(?!\[/E\])', f"[E]{entity}[/E]", sentence)

            if found_entities:
                entity_list = ",".join(found_entities)  # 逗号分隔的实体列表
                entity_count = len(found_entities)  # 计算实体数量
                if entity_count >= 2:
                    print(sentence)
                    f_out.write(f"{sentence}\t{entity_list}\t{entity_count}\n")  # 追加信息

# 文件路径
entities_path = "filtered_entities.txt"
sentences_path = "sentences.txt"
output_path = "filtered_sentences.txt"

# 处理数据
entities = load_entities(entities_path)
annotate_sentences(sentences_path, entities, output_path)

print("filtered_sentences.txt 生成完毕！")
