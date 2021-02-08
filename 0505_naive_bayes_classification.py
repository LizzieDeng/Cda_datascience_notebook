"""
朴素贝叶斯分类算法（Naive Bayes Classification）：
P(A/B) = p(A)*P(B/A)/P(B)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB


sns.set()
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()

model = GaussianNB()
model.fit(X, y)
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.show()

yprob = model.predict_proba(Xnew)
print(yprob[-8:].round(2))

"""
Multinomial Navie Bayes: 多项式朴素贝叶斯
这个方法假定数据的特征是从一个简单的多项式分布中生成的，多项式分布描述了在一些分组中观察到的计数的概率，因此多项式朴素贝叶斯对于表达技术或技术的比例之类的特征还是最适合的，
分类文字（Classifying Text）: 多项式朴素贝叶斯经常被用到的场合是文字分类，这种场景下的特征是单词的计数或者文档中单词出现的频率。
"""

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
print(data.target_names)

categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])
# 为了将这个数据集应用到机器学习上，我们需要将数据中的每个字符串内容转换成数字的向量，使用TF-IDF来实现向量化，然后创建一个管道操作将一个多项式朴素贝叶斯分类器连接进来：
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# 我们可以将这个管道应用到训练集上，然后在测试集上去进行标签预测
model.fit(train.data, train.target)
labels = model.predict(test.data)
# 有了对测试数据预测的标签之后，我们可以对评估器的性能做出判断
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names,
            yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predict label')
plt.show()


# 创建一个简单的工具函数来对任何字符串输入返回标签预测的输出结果
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


print(predict_category('send a payload to the ISS'))
print(predict_category('discussing islam vs atheism'))
print(predict_category('determining the screen resolution'))



