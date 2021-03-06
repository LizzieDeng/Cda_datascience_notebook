#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：Cda_datascience_notebook -> machineLearning
@IDE    ：PyCharm
@Author ：LizzieDeng
@Date   ：2021/1/9 13:01
@Desc   ：machineLearning notebook
===================================================="""
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# common plot formatting
def format_plot(ax, title):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xlabel('feature 1', color='gray')
    ax.set_ylabel('feature 2', color='gray')
    ax.set_title(title, color='gray')


# creatw 50 separable points
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)

# fit the support vector classifier model
clf = SVC(kernel='linear')
clf.fit(X, y)

# create some new points to predict
X2, _ = make_blobs(n_samples=80, centers=2, random_state=0, cluster_std=0.8)
X2 = X2[50:]

# predict the labels
y2 = clf.predict(X2)
point_style = dict(cmap='Paired', s=50)
line_style = dict(levels=[-1, 0, 1], linestyles=['dashed', 'solid', 'dashed'], colors='gray', linewidths=1)


# ########### classification example figure 1 #################
def plot_figure_1():
    # plot the data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, **point_style)

    # format plot
    format_plot(ax, 'Input Data')
    ax.axis([-1, 4, -2, 7])
    plt.show()
    fig.savefig('figures/05.01-classification-1.png')


# ##################### Classification Figure 2 #############
# get contour describing the model
def plot_figure_2():
    xx = np.linspace(-1, 4, 10)
    yy = np.linspace(-2, 7, 10)
    xy1, xy2 = np.meshgrid(xx, yy)
    Z = np.array([clf.decision_function([t]) for t in zip(xy1.flat, xy2.flat)]).reshape(xy1.shape)
    # print('Z:', xy1.shape, xy2)
    # plot points and model
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, **point_style)
    ax.contour(xy1, xy2, Z, **line_style)

    # format plot
    format_plot(ax, 'Model Learned from Input Data')
    ax.axis([-1, 4, -2, 7])
    plt.show()
    fig.savefig('figures/05.01-classification-2.png')


def plot_figure_3():
    # plot the results
    xx = np.linspace(-1, 4, 10)
    yy = np.linspace(-2, 7, 10)
    xy1, xy2 = np.meshgrid(xx, yy)
    Z = np.array([clf.decision_function([t]) for t in zip(xy1.flat, xy2.flat)]).reshape(xy1.shape)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    ax[0].scatter(X2[:, 0], X2[:, 1], c='gray', **point_style)
    ax[0].axis([-1, 4, -2, 7])
    ax[1].scatter(X2[:, 0], X2[:, 1], c=y2, **point_style)
    ax[1].contour(xy1, xy2, Z, **line_style)
    ax[1].axis([-1, 4, -2, 7])
    format_plot(ax[0], 'Unknown Data')
    format_plot(ax[1], 'Predicted Labels')
    plt.show()
    fig.savefig('figures/05.01-classification-3.png')


def regression_learning():
    # create some data for the regression
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    y = np.dot(X, [-2, 1] + 0.1 * rng.randn(X.shape[0]))

    # fit the regression model
    model = LinearRegression()
    model.fit(X, y)

    # create some new points to predict
    X2 = rng.randn(100, 2)

    # predict the labels
    y2 = model.predict(X2)

    # plot data points
    fig, ax = plt.subplots()
    points = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')

    # format plot
    format_plot(ax, "Input Data")
    ax.axis([-4, 4, -3, 3])
    fig.savefig('figures/05.01-regression-1.png')


def regression_data_figure_2():
    points = np.hstack([X, y[:, None]]).reshape(-1, 1, 3)
    segments = np.hstack([points, points])
    segments[:, 0, 2] = -8

    # plot points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c=y, s=35, cmap='viridis')
    ax.add_collection3d(Line3DCollection(segments, colors='gray', alpha=0.2))
    ax.scatter(X[:, 0], X[:, 1], -8+np.zeros(X.shape[0]), c=y, s=10, cmap='viridis' )

    # format plot
    ax.patch.set_facecolor('white')
    ax.view_init(elev=20, azim=-70)
    ax.set_zlim3d(-8, 8)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())
    ax.set(xlabel='feature 1', ylabel='feature 2', zlabel='label')

    # Hide axes(is there a better way?)
    ax.w_xaxis.line.set_visible(False)
    ax.w_yaxis.line.set_visible(False)
    ax.w_zaxis.line.set_visible(False)
    for tick in ax.w_xaxis.get_ticklines():
        tick.set_visible(False)
    for tick in ax.w_yaxis.get_ticklines():
        tick.set_visible(False)
    for tick in ax.zaxis.get_ticklines():
        tick.set_visible(False)
    plt.show()
    fig.savefig('figures/05.01-regression-2.png')


def learing_sklearn():
    iris = sns.load_dataset('iris', cache=True, data_home="./seaborn-data")
    print(iris.head())
    sns.set()
    sns.pairplot(iris, hue='species', size=1.5)
    plt.show()
    X_iris = iris.drop('species', axis=1)
    print(X_iris.shape)
    y_iris = iris['species']
    print(y_iris.shape)
    rng = np.random.RandomState(42)
    x = 10 * rng.rand(50)
    y = 2 * x - 1 + rng.randn(50)
    plt.scatter(x, y)
    plt.show()
    # 1. choose a class of model
    # 2. choose model hyperparameters
    model = LinearRegression(fit_intercept=True)
    print("model : {}s".format(model))
    # 3. arange data into a features matrix and target vector
    X = x[:, np.newaxis]
    print(X.shape)
    # 4. fit the model to your data
    model.fit(X, y)
    print(model.coef_)
    print(model.intercept_)
    # 5. predict labels for unknown data
    xfit = np.linspace(-1, 11)
    Xfit = xfit[:, np.newaxis]
    print('xfit', xfit, 'Xfit', Xfit)
    print(xfit.shape, Xfit.shape)
    yfit = model.predict(Xfit)
    plt.scatter(x, y)
    plt.plot(xfit, yfit)
    plt.show()


def iris_classification():
    iris = sns.load_dataset('iris', cache=True, data_home="./seaborn-data")
    X_iris = iris.drop('species', axis=1)
    y_iris = iris['species']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)
    from sklearn.naive_bayes import GaussianNB    # 1. choose model class
    model = GaussianNB()                          # 2. instantiate model
    model.fit(Xtrain, ytrain)                     # 3. fit model to data
    y_model = model.predict(Xtest)                # 4. predict on new data
    from sklearn.metrics import accuracy_score
    ac_src = accuracy_score(ytest, y_model)
    print(ac_src)


iris = sns.load_dataset('iris', cache=True, data_home="./seaborn-data")
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']


def unsupervised_learning_example():
    from sklearn.decomposition import PCA   # 1. choose the model class
    model = PCA(n_components=2)             # 2. instantiate model
    model.fit(X_iris)                       # 3. Fit to data. Notice y is not specified!
    X_2D = model.transform(X_iris)          # 4. Transform the data to two dimensions
    iris['PCA1'] = X_2D[:, 0]
    iris['PCA2'] = X_2D[:, 1]
    sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)
    plt.show()


def unsupervised_learning():
    from sklearn.mixture import GaussianMixture                     # 1. choose the model class
    model = GaussianMixture(n_components=3, covariance_type='full') # 2. Instantiate the model with hyperparmeter
    model.fit(X_iris)         # 3. fit to data. Notice y is not specified
    y_gmm = model.predict(X_iris) # 4. determine cluster labels
    iris['cluster'] = y_gmm
    sns.lmplot('PCA1', 'PCA2', data=iris, hue='species', col='cluster', fit_reg=False)

















if __name__ == '__main__':
    # plot_figure_3()
    # regression_data_figure_2()
    # learing_sklearn()
    # iris_classification()
    # unsupervised_learning_example()
    unsupervised_learning()
