def process_texts(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w+", encoding="utf-8") as f_out:
        for line in f_in:
            parts = line.strip().split("\t")  # 假设用制表符分隔
            if len(parts) < 2:
                continue  # 忽略不符合格式的行

            text = parts[1]  # 取出第二列（长文本）
            sentences = text.split("。")  # 按句号分割

            # 写入分割后的句子
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:  # 过滤掉空句子
                    f_out.write(sentence + "\n")


# 文件路径
input_path = "new_long_texts.txt"
output_path = "sentences.txt"

# 处理数据
process_texts(input_path, output_path)

print("sentences.txt 生成完毕！")
