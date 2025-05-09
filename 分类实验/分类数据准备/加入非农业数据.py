import json


# 读取文件，每行作为列表元素
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


# 处理类别 4 的数据并生成 JSON
def process_non_agriculture_data(non_agriculture_data, output_file):
    results = []

    for line in non_agriculture_data:
        label, text = line.split("\t", 1)

        # 只有类别 4 才处理
        if int(label) == 4:
            news_data = {
                "Title": text,
                "Entity": '',
                "gra": '',
                "Label": int(label)
            }
            results.append(news_data)

    # 将数据追加到现有的 JSON 文件中
    try:
        with open(output_file, 'r+', encoding='utf-8') as f:
            # 读取现有内容
            existing_data = json.load(f)
            # 追加新数据
            existing_data.extend(results)
            # 重新写入
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        # 如果文件不存在，则直接创建并写入
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"类别 4 数据处理完成，已保存到 {output_file}")


# 主程序
def main():
    non_agriculture_data_path = '/home/ypx/project/ypx/Agriculture_projects/实验数据集/非农业数据集/title.txt'  # 类别 4 的数据文件
    output_json_path = '/home/ypx/project/ypx/Agriculture_projects/分类实验/datas/Generate_KG/GeneraKG.json'

    # 读取类别 4 的数据
    non_agriculture_data = read_file(non_agriculture_data_path)

    # 处理并将类别 4 数据追加到 JSON 文件
    process_non_agriculture_data(non_agriculture_data, output_json_path)


if __name__ == "__main__":
    main()
