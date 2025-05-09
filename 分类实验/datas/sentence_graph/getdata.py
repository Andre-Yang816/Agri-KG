import json

# 读取 JSON 文件
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载数据
json1 = load_json('/home/ypx/project/ypx/Agriculture_projects/分类实验/datas/non_agri/processed_news.json')
json2 = load_json('/home/ypx/project/ypx/Agriculture_projects/分类实验/datas/gra/expanded_data.json')

# 确保两个 JSON 文件行数一致
assert len(json1) == len(json2), "两个 JSON 文件的行数不匹配"

# 合并数据
merged_data = []
for item1, item2 in zip(json1, json2):
    merged_item = {
        "Title": item1["Title"],
        "Entity": item1["Entity"],
        "Label": item1["Label"],
        "Sentence": item1["Sentence"],
        "gra": item2["gra"]
    }
    merged_data.append(merged_item)

# 保存合并后的数据
with open('sentence_graph.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print("合并完成，结果保存在 merged.json")