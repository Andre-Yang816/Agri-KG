

import random
import re

# 读取1.txt和2.txt文件
with (open('/BertClassifier-master/datasets/entities/long_texts_entities.txt', 'r', encoding='utf-8') as f1,
      open('/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/datasets/sentences/sentences.txt', 'r', encoding='utf-8') as f2):
    lines_1 = f1.readlines()
    lines_2 = f2.readlines()



# 检查并保留有效行
valid_lines_1 = []
valid_lines_2 = []

for line_1, line_2 in zip(lines_1, lines_2):
    parts_1 = line_1.strip().split('\t')
    if len(parts_1) == 2 and parts_1[1]:  # 如果第二列不为空
        valid_lines_1.append(line_1)
        #这里对line_2处理
        texts = ''.join(re.findall(r'[\u4e00-\u9fff\u3001-\u303F\uFF00-\uFFEF0-9]', line_2))[1:]
        texts = line_2[0] + '\t' + texts + '\n'
        valid_lines_2.append(texts)



with open('./new_sentences.txt', 'w', encoding='utf-8') as f2_out:
    f2_out.writelines(valid_lines_2)
