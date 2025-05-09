import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


# 1. 数据集处理
class TextDataset(Dataset):
    def __init__(self, filename, word2vec_model, vectorizer):
        self.texts = []
        self.labels = []
        self.vectorizer = vectorizer
        self.word2vec = word2vec_model
        self.load_data(filename)

    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        texts, labels = [], []
        for line in lines:
            label, text = line.strip().split('\t')
            labels.append(int(label))
            texts.append(text)

        self.tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        self.w2v_features = np.array([self.text_to_w2v(text) for text in texts])
        self.labels = np.array(labels)

    def text_to_w2v(self, text):
        words = text.split()
        word_vectors = [self.word2vec.wv[word] for word in words if word in self.word2vec.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(self.word2vec.vector_size)

    def __getitem__(self, index):
        return np.concatenate((self.tfidf_features[index], self.w2v_features[index])), self.labels[index]

    def __len__(self):
        return len(self.labels)


# 2. CNN 分类模型
class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(input_size // 2 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, feature_dim)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 3. 训练模型
def train_model(train_loader, model, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.float(), labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')


# 4. 主程序
if __name__ == '__main__':
    train_file = 'train.txt'  # 你的数据文件
    word2vec_model = Word2Vec.load('word2vec.model')  # 加载预训练 Word2Vec
    vectorizer = TfidfVectorizer(max_features=5000)

    dataset = TextDataset(train_file, word2vec_model, vectorizer)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNNClassifier(input_size=5000 + word2vec_model.vector_size, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, model, criterion, optimizer)
