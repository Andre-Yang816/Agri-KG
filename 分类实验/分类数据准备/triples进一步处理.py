import csv

# 读取输入 CSV 文件并处理
input_filename = 'matched_triples.csv'  # 输入文件名
output_filename = 'new_matched_triples.csv'  # 输出文件名

# 读取 CSV 文件中的数据
input_data = []
with open(input_filename, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过表头
    for row in reader:
        input_data.append((row[0], row[1], row[2]))

# 处理后的结果
output_data = []

# for entity1, relation, entity2 in input_data:
#     if relation == "包含":
#         # 对包含关系进行互换，并替换关系
#         output_data.append((entity2, "属于", entity1))
#     elif relation == "危害":
#         # 对危害关系进行互换，并替换关系
#         output_data.append((entity2, "被危害", entity1))
#     else:
#         # 其他关系保持不变
#         output_data.append((entity1, relation, entity2))
#
# # 输出处理后的数据到 CSV 文件
# with open(output_filename, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Entity1", "Relation", "Entity2"])  # 写入表头
#     writer.writerows(output_data)

# print(f"处理结果已保存到 {output_filename}")

list_1 = []
for entity1, relation, entity2 in input_data:
    if len(entity1) == 1:
        list_1.append(entity1)
    if len(entity2) == 1:
        list_1.append(entity1)
print(list_1)