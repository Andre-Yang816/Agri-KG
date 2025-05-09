import random


# 读取txt文件的内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines


# 将数据写入文件
def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(data)


# 按比例划分数据集
def split_data(lines, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    total_lines = len(lines)
    train_size = int(total_lines * train_ratio)
    val_size = int(total_lines * val_ratio)
    test_size = total_lines - train_size - val_size

    random.shuffle(lines)

    train_data = lines[:train_size]
    val_data = lines[train_size:train_size + val_size]
    test_data = lines[train_size + val_size:]

    return train_data, val_data, test_data


# 主程序
def main():
    input_file = 'whole_sentences.txt'  # 输入文件路径
    train_file = 'train_sentences.txt'  # 训练集文件路径
    val_file = 'val_sentences.txt'  # 验证集文件路径
    test_file = 'test_sentences.txt'  # 测试集文件路径

    # 读取文件内容
    lines = read_file(input_file)

    # 划分数据
    train_data, val_data, test_data = split_data(lines)

    # 将划分后的数据写入到不同文件
    write_to_file(train_file, train_data)
    write_to_file(val_file, val_data)
    write_to_file(test_file, test_data)

    print(f"数据集已划分为：\n训练集：{len(train_data)} 条\n验证集：{len(val_data)} 条\n测试集：{len(test_data)} 条")


if __name__ == '__main__':
    main()
