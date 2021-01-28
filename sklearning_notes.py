#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：Cda_datascience_notebook -> sklearning_notes
@IDE    ：PyCharm
@Author ：LizzieDeng
@Date   ：2021/1/12 20:17
@Desc   ：application exploring hand-written digits
=================================================="""
from sklearn.datasets import load_digits, load_iris
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def introduce_sklearn():
    digits = load_digits()
    print(digits.images.shape)
    print(digits.images[:10, :])

    # plot 100 records in load_digits data
    fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        # print(i, ax)
        ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
        ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')
    # plt.show()
    X = digits.data
    print(X.shape)
    # print(digits)
    y = digits.target
    print(y.shape)


def hyperparameters_and_model_validation():
    """
    监督机器学习模型基本操作:
    1. 选择一个模型类
    2. 选择模型参数
    3. 训练数据拟合模型
    4.利用模型预测新数据的性质

    思考验证模型：选择好模型类和参数时，可以通过使用一些训练数据级或者将预测值与实际值比较
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)
    y_model = model.predict(X)
    ac_src = accuracy_score(y, y_model)
    print('ac_src is {}'.format(ac_src))
    # 即使准确因子1.0意味着100%的点与我们的模型匹配，但是这并不意味着准确度高，因为这个模型有个缺陷：
    # 它是用同样的数据来训练和推测，最近邻进算法是一个即时预测器，只是简单的训练数据，
    # 将新数据与这些存储的数据进行比较并预测属性，除了认为情况，否则模型每次都会得到100%的准确率
    # 那么该如何做？
    # 模型验证的正确方法：Holdout sets， 保留数据集（训练数据集的子集不参与算法训练而用于验证算法）
    from sklearn.model_selection import train_test_split
    # split the data with 50% in each set
    X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)
    # fit the model on one set of data
    model.fit(X1, y1)

    # evaluate the model on the second set of data
    y2_model = model.predict(X2)
    accua = accuracy_score(y2, y2_model)
    print(accua)
    # 交叉验证（cross-validation）方法验证模型：将数据集分为均等不相交的K份，然后取其中一份进行测试，另外的k-1进行训练，然后求得error的平均值作为最终的评价
    # 保留数据集的缺点在于我们丢失了部分训练数据集，特别是当训练集数据量较小的时候
    y22_model = model.fit(X1, y1).predict(X2)
    y11_model = model.fit(X2, y2).predict(X1)
    print(y11_model, y1)
    print(y22_model, y2)
    # 将数据分成5组，将每组与其他4组进行测评，使用cross_val_score
    from sklearn.model_selection import cross_val_score
    print(cross_val_score(model, X, y, cv=5))
    # leave one out cross validation
    # 该方法是将数据集分为训练集和数据集，但是测试集只有一个数据，其他的数据都作为训练集，并将此步骤重复N次
    from sklearn.model_selection import LeaveOneOut
    print("len(X) is {}".format(len(X)))
    scores = cross_val_score(model, X, y, cv=LeaveOneOut())
    print("scores is {}".format(scores))
    print("score's mean is {}".format(scores.mean()))










if __name__ == '__main__':
    hyperparameters_and_model_validation()



