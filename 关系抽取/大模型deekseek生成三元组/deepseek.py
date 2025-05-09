import json

import requests
import itertools

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

url = "https://api.siliconflow.cn/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-gdfwxagfrusrgvlmstfvskrzschcrjhkoyirtnyjzaqadyqy",
    "Content-Type": "application/json"
}

# Relationship dictionary (for filtering GPT's generated relations)
relation_dict = ["属于", "防治", "别名", "感染", "危害", "包含", "病原", "目", "科", "纲"]

def extract_relations(entity1, entity2, sentence):
    prompt = f"""
        你是农业知识专家，正在构建农业知识图谱,你需要：
        1. 分析实体间的所有可能关系，仅限于农业领域
        2. 使用以下标准关系词之一（仅限于以下选项）："属于", "防治", "别名", "感染", "被危害", "导致", "病原", "目", "科", "纲", "不良反应"
        3. 以严格标准格式，返回句子中的关系，不要额外的文字说明或解释
        4. 如果认为不存在关系则返回None
        请分析以下句子中的农业实体对的可能关系，句子：{sentence},句子中的实体1：{entity1},实体2：{entity2}，
        请返回预测的关系：
        """
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "messages": [
            {"role": "system", "content": "你是一个农业知识专家，帮助用户构建农业知识图谱。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024,

    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        result = result["choices"][0]["message"]["content"]
        pred_relation = result.split(',')[-1]
        print(result)
        print(pred_relation)
        return pred_relation
    else:
        print("Error:", response.text)
        return None


# 评估函数
def evaluate(predictions, ground_truth):
    # 计算准确率、精确率、召回率、F1值
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='micro')
    recall = recall_score(ground_truth, predictions, average='micro')
    f1 = f1_score(ground_truth, predictions, average='micro')

    return accuracy, precision, recall, f1


# 加载数据
with open('/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/评估/eva_data0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 存储真实标签与预测标签
true_labels = []
predicted_labels = []

# 存储结果数据
result_data = []
# 逐条数据进行关系预测
for entry in data:
    text = entry['句子']
    true_relation = entry['关系'][0]  # 假设每个样本只有一个关系
    entity1 = entry['实体1']
    entity2 = entry['实体2']

    predicted_relation = extract_relations(entity1, entity2, text)
    if predicted_relation == None:
        break
    # 将结果存入字典，格式为实体1, 实体2, 真实关系, 预测关系
    result_data.append({
        "实体1": entity1,
        "实体2": entity2,
        "真实关系": true_relation,
        "预测关系": predicted_relation
    })
    # 对比预测与真实关系
    if true_relation in predicted_relation:
        predicted_labels.append(1)  # 预测正确
    else:
        predicted_labels.append(0)  # 预测错误

    true_labels.append(1)  # 每条数据的真实标签为1

# 评估结果
accuracy, precision, recall, f1 = evaluate(predicted_labels, true_labels)

# 打印评估结果
print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1值: {f1:.4f}")

with open('prediction_results.json', 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)