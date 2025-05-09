import json
import re


# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


# 提取新闻标题从新闻全文中删除
def remove_title_from_text(title, full_text):
    # 假设标题与全文开头一致，直接去除标题
    if full_text.startswith(title):
        return full_text[len(title):].strip()
    return full_text.strip()


# 读取实体信息
def load_entities(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(f.read().strip().split('\n'))


# 查找新闻中的实体
def extract_entities_from_text(text, entities):
    found_entities = []
    for entity in entities:
        if entity in text:
            found_entities.append(entity)
    return found_entities


# 获取相关句子
def get_relevant_sentences(text, entities):
    sentences = text.split(r'(?<=[。！？])')  # 按句号分割
    relevant_sentences = []
    for sentence in sentences:
        for entity in entities:
            if entity in sentence:
                relevant_sentences.append(sentence.strip())
                break
    return relevant_sentences


# 处理每篇新闻并生成 JSON
def process_news(new_long_texts, new_shortText, entities, output_file):
    results = []
    for i in range(len(new_long_texts)):
        label, full_text = new_long_texts[i].strip().split("\t", 1)
        title = new_shortText[i].strip().split('\t')[1]

        # 去除新闻标题
        cleaned_text = remove_title_from_text(title, full_text)

        # 提取新闻中的实体
        # found_entities = extract_entities_from_text(cleaned_text, entities)
        found_entities = [entity for entity in entities if entity in cleaned_text]

        # 获取相关句子
        # relevant_sentences = get_relevant_sentences(cleaned_text, found_entities)
        sentences = re.split(r'(?<=[。！？])', full_text)  # 按句号、问号、叹号分割句子
        related_sentences = [s for s in sentences if any(e in s for e in found_entities)]

        # 生成 JSON 数据
        news_data = {
            "Title": title,
            "Entity": ",".join(found_entities),
            "Sentence": " ".join(related_sentences),
            "Label": int(label)
        }

        results.append(news_data)

    # 输出为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# 主程序
def main():
    new_long_texts_path = 'new_long_texts.txt'
    new_shortText_path = 'new_shortText.txt'
    unique_entities_path = 'unique_entities.txt'
    output_json_path = 'processed_news.json'

    # 读取文件
    new_long_texts = read_file(new_long_texts_path)
    new_shortText = read_file(new_shortText_path)
    entities = load_entities(unique_entities_path)

    # 处理数据并生成 JSON
    process_news(new_long_texts, new_shortText, entities, output_json_path)
    print(f"处理完成，结果已保存到 {output_json_path}")


# 运行主程序
if __name__ == "__main__":
    main()
