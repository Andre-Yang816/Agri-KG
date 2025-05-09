import json
import random
from collections import defaultdict


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def classify_by_relation(data, given_relations):
    relation_dict = {relation: [] for relation in given_relations}
    for item in data:
        for relation in item["关系"]:
            if relation in relation_dict:
                relation_dict[relation].append(item)
    return relation_dict


def sample_and_save(relation_dict, num_samples=13, num_iterations=10, output_prefix="eva_data"):
    for i in range(num_iterations):
        sampled_data = []
        for relation, examples in relation_dict.items():
            if len(examples) >= num_samples:
                sampled_data.extend(random.sample(examples, num_samples))
            else:
                sampled_data.extend(examples)  # 若样本不足，全部加入

        output_file = f"{output_prefix}{i}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=4)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    input_file = "/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/data/test_data.json"  # 请替换成你的数据文件
    given_relations = ["防治", "感染", "别名", "被危害", "科", "导致", "病原", "目", "纲", "", "", ""]  # 请替换成你的固定关系列表

    data = load_data(input_file)
    relation_dict = classify_by_relation(data, given_relations)
    sample_and_save(relation_dict)
