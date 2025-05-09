import re
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 数据预处理（适用于中文）
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = ' '.join(jieba.cut(text))  # 使用 jieba 进行中文分词
    return text

# 2. 加载数据
def load_data(filename):
    texts, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        label, text = line.strip().split('\t')
        texts.append(clean_text(text))  # 预处理中文文本
        labels.append(int(label))
    return texts, labels

# 3. 处理文本数据（TF-IDF + Trigrams）
def process_data(texts):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)  # 1-3元 N-gram
    tfidf_features = vectorizer.fit_transform(texts)
    return tfidf_features, vectorizer

# 4. 训练 LinearSVC 模型
def train_svm(X_train, y_train):
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model

# 5. 评估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred,digits=4))

# 6. 主程序
if __name__ == '__main__':
    train_path = '/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/datasets/longText/train_texts.txt'
    val_path = '/home/ypx/project/ypx/Agriculture_projects/BertClassifier-master/datasets/longText/test_texts.txt'

    # 读取训练数据
    X_1, y_train = load_data(train_path)
    X_train, vectorizer = process_data(X_1)

    # 读取测试数据
    X_2, y_test = load_data(val_path)
    X_test = vectorizer.transform(X_2)  # 使用训练时的 vectorizer 进行 transform

    # 训练模型
    model = train_svm(X_train, y_train)

    # 评估模型
    evaluate_model(model, X_test, y_test)
