from collections import defaultdict


def count_chars_by_label(file_path):
    label_char_count = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue  # 跳过格式错误的行
            label, text = parts
            label_char_count[label] += len(text)

    return label_char_count


if __name__ == "__main__":
    file_path = "whole_texts.txt"
    char_counts = count_chars_by_label(file_path)

    for label, count in char_counts.items():
        print(f"类别: {label}, 字符总数: {count}")
