# 脚本功能：检查两个文件的行数是否相等，若相等，则将第一个文件的第二列附加在第二个文件的第二列之前。

def check_and_merge(file1, file2, output_file):
    # 打开文件并读取数据
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    print(len(lines1))
    print(len(lines2))
    # 检查文件行数是否相等
    if len(lines1) != len(lines2):
        print("Error: The number of lines in the files is not equal.")
        return

    merged_lines = []
    for line1, line2 in zip(lines1, lines2):
        # 分割每行的数据
        parts1 = line1.strip().split('\t')  # 假设文件1的每行是由Tab分隔的
        parts2 = line2.strip().split('\t')  # 假设文件2的每行是由Tab分隔的

        # 将文件1的第二列附加在文件2的第二列之前
        if len(parts1) >= 2:
            if len(parts2) >= 2:
                new_line = parts2[0] + '\t' + parts1[1] + '，' + parts2[1]
            else:
                new_line = parts2[0] + '\t' + parts1[1]
        else:
            print(f"Warning: Invalid line format in files: {line1}, {line2}")
            continue

        merged_lines.append(new_line)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as out:
        for line in merged_lines:
            out.write(line + '\n')

    print(f"Merge successful! The result is saved in {output_file}")


# 使用示例
file1 = '/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/datasets/shortText/new_shortText.txt'  # 第一个文件路径

file2 = '/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/datasets/entities/new_long_texts_entities.txt'  # 第二个文件路径
output_file = 'merged_entities.txt'  # 输出文件路径

check_and_merge(file1, file2, output_file)
