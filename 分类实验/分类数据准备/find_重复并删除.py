from collections import defaultdict


def find_duplicate_line_numbers(short_text_file):
    title_dict = defaultdict(list)  # 用于存储标题对应的行号

    with open(short_text_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t', 1)  # 按制表符分割
            if len(parts) == 2:
                _, title = parts
                title_dict[title].append(idx)

    # 找到所有重复的行号
    duplicate_line_numbers = {idx for indices in title_dict.values() if len(indices) > 1 for idx in indices}
    return duplicate_line_numbers


def remove_lines_by_index(input_file, output_file, line_numbers):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = [line for idx, line in enumerate(lines) if idx not in line_numbers]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)


def process_files(short_text_file, long_text_file):
    duplicate_line_numbers = find_duplicate_line_numbers(short_text_file)

    remove_lines_by_index(long_text_file, long_text_file, duplicate_line_numbers)
    remove_lines_by_index(short_text_file, short_text_file, duplicate_line_numbers)

    print("处理完成，重复标题的行已从两个文件中删除。")


# 文件路径
new_long_texts_file = "new_long_texts.txt"
new_short_texts_file = "new_shortText.txt"

process_files(new_short_texts_file, new_long_texts_file)
