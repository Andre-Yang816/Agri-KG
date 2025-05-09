import matplotlib.pyplot as plt
import numpy as np

# 实验结果数据
experiment_numbers = [0, 1, 2, 3, 4]
bert_precision = [0.81517, 0.76745, 0.73481, 0.75234, 0.75304]
bert_recall = [0.77011, 0.70103, 0.68376, 0.74766, 0.7766]
bert_f1 = [0.77901, 0.71158, 0.67841, 0.74154, 0.74915]

model2_precision = [0.64, 0.64, 0.41, 0.77, 0.75]
model2_recall = [0.57, 0.5, 0.53, 0.46, 0.56]
model2_f1 = [0.51, 0.47, 0.45, 0.4, 0.53]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(experiment_numbers, bert_precision, label="BERT Precision", marker='o')
plt.plot(experiment_numbers, bert_recall, label="BERT Recall", marker='o')
plt.plot(experiment_numbers, bert_f1, label="BERT F1", marker='o')

plt.plot(experiment_numbers, model2_precision, label="model2 Precision", marker='s')
plt.plot(experiment_numbers, model2_recall, label="model2 Recall", marker='s')
plt.plot(experiment_numbers, model2_f1, label="model2 F1", marker='s')

# 添加标题和标签
plt.title('Model Comparison: BERT-BiLSTM vs model2')
plt.xlabel('Experiment Number')
plt.ylabel('Scores')
plt.legend()

# 显示图形
plt.grid(True)
plt.tight_layout()
plt.show()
