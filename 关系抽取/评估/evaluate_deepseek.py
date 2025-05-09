import json
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import logging

log_dir = 'logs/deepSeekv3.log'

# Set up logging
logging.basicConfig(filename=log_dir, level=logging.INFO)

# Example data (replace with your actual data)
with open('/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/DeepSeekV3/pred4.json', 'r', encoding='utf-8') as f:
    pred_data = json.load(f)

with open('/home/ypx/project/ypx/Agriculture_projects/关系抽取任务/评估/eva_data4.json', 'r', encoding='utf-8') as f:
    true_data = json.load(f)

label_map = {
    "属于": 0, "防治": 1, "感染": 2, "别名": 3, "被危害": 4, "科": 5, "导致": 6, "病原": 7,
    "目": 8, "纲": 9, "不良反应": 10, '没有关系': 13
}

true_labels = []
pred_labels = []
logging.info('test number 4')
# Convert the relationship to a list of labels for comparison
for i in range(len(true_data)):

    if true_data[i]["实体1"] == pred_data[i]["实体1"] and true_data[i]["实体2"] == pred_data[i]["实体2"]:
        true_labels.append(label_map[true_data[i]["关系"][0]])
        pred_labels.append(label_map[pred_data[i]["关系"]])
    else:
        print(f"Mismatch at index {i}:")
        print("True data:", true_data[i])
        print("Pred data:", pred_data[i])
        continue

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Identify the set of labels actually present in the true and predicted data
unique_labels = list(set(true_labels + pred_labels))

# Calculate precision, recall, and F1 score
report = classification_report(true_labels, pred_labels, labels=unique_labels, target_names=[k for k in label_map.keys()], output_dict=True, digits=4)

# Log the classification report for each class
logging.info("****DeepSeek V3 Prediction Evaluation Report****")
logging.info('\n' + classification_report(true_labels, pred_labels, labels=unique_labels, target_names=[k for k in label_map.keys()]))

# Optionally, you can print the report to the console as well
print("****DeepSeek V3 Prediction Evaluation Report****")
print(classification_report(true_labels, pred_labels, labels=unique_labels, target_names=[k for k in label_map.keys()]))
