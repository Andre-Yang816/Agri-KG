# 打开1.txt和2.txt文件，读取内容
with open('merged_entities.txt', 'r', encoding='utf-8') as file1, open('title.txt', 'r', encoding='utf-8') as file2:
    lines1 = file1.readlines()  # 读取1.txt的所有行
    lines2 = file2.readlines()  # 读取2.txt的所有行

# 将1.txt和2.txt的内容合并
lines = lines1 + lines2

# 写入合并后的内容到3.txt文件
with open('whole_entities.txt', 'w', encoding='utf-8') as file3:
    file3.writelines(lines)

print("文件内容已成功合并并写入到whole_entities！")
