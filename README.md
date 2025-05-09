# AKG & AKG-BERT

## 数据集
数据集包含：农业新闻文本分类语料库（标题、全文、标注类别）、农业实体集、知识图谱

在“/datasets"目录下存放着最终的所有数据集，这些数据集可用于支撑相关研究。

其中，NERT.txt 用于命名实体识别任务的训练，filtered_entities.txt为最终获取的实体集合。

final_relations.csv为最终获取的三元组集

processed_news.zip为本文涉及的所有语料信息，并进行了初步预处理。

在所有模型的文件夹下，仍然有对应用于训练的数据集。


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
