# AKG & AKG-BERT

## 数据集
数据集包含：农业新闻文本分类语料库（标题、全文、标注类别）、农业实体集、知识图谱

## 模型代码
包含：
（1）NER任务的所有训练模型（BERT/BERT-CRF/BERT-BiLSTM-CRF/ERNIE-CRF/ERNIE-BiLSTM-CRF/Weighted-ERNIE-BiLSTM-CRF）

（2）关系抽取代码，包含大模型deepseek的api调用代码

（3）多通道信息融合AKG-BERT代码

（4）评估代码

（5）制图代码
 
## 文件分支介绍
“/AW-ERNIE”至“/ERNIE-KAN-W-CRF”文件夹分别为构建指示图谱中的命名实体识别任务涉及的各模型的网络结构、参数配置等，每一个模型都是完全独立的；

 “/关系抽取”是关系抽取任务的全部数据和代码；
 
 “/分类实验”文件夹包含*文本分类*任务的所有代码。
