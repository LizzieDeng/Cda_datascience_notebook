# -*- coding: utf-8 -*-
"""
@Time: 2021/2/8 12:26
@Author: LizzieDeng
@File: 0506_Linear_Regression.py
@Description:
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
sns.set()
"""
简单线性回归： Simple Linear Regression
y = ax + b
a 通常被称为斜率，而b通常被成为截距。
"""

# rng = np.random.RandomState(1)
# x = 10 * rng.rand(50)
# y = 2 * x - 5 + rng.randn(50)
# plt.scatter(x, y)
# plt.show()

# 使用Scikit_Learn的linerRegression 评估其来拟合这些数据然后得到一条最佳拟合直线

# model = LinearRegression(fit_intercept=True)
# model.fit(x[:, np.newaxis], y)
# xfit = np.linspace(0, 10, 1000)
# yfit = model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.show()
# 数据的斜率和截距可以在模型拟合参数中找到，在Scikit-Learn中总是使用下划线后缀来表示。这里相关的参数是coef_和intercept_：
# print("Model slope    :", model.coef_[0])
# print("Model intercept:", model.intercept_)

# LinearRegression评估器能做的远不止于此，除了简单的直线拟合外，它还能处理多维线性模型的形式
# y=a0+a1x1+a2x2+⋯
# 这里有多个x值。几何上，这等同于在三维空间间使用一个平面拟合数据，或在更高维空间中使用超平面拟合数据。
# 这样的回归具有多维的本质，因此令它们比较难以可视化，但我们可以构造一些样例数据来查看这样的拟合，这里使用了NumPy的矩阵乘法操作：

# rng = np.random.RandomState(1)
# X = 10 * rng.rand(100, 3)
# y = 0.5 + np.dot(X, [1.5, -2, 1])
# model.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# 这里y值是由三个随机x值构建的，而线性回归恢复了用来构建数据的斜率。 使用这种方法，我们可以使用单个LinearRegression评估器拟合直线、平面或超平面到数据上。目前为止这种方法看起来都限制在变量之间的线性关联上，但是实际上它还能完成更多的工作。

"""
基本函数回归：Basic Function Regression
将线性回归应用在变量之间的非线性关系的一个技巧是，将数据通过基本函数进行转换。我们在超参数和模型验证和特征工程中已经看到过多项式回归PolynomialRegression管道操作中已经看到这个技巧的例子。这个方法是将一维的输入数据使用多维线性模型
y=a0+a1x1+a2x2+a3x3+⋯
来建立x1,x2,x3等。即我们令xn=fn(x)其中的fn()是用来转换数据的函数。
例如，如果令fn(x)=xn，我们的模型就会变成一个多项式回归：
y=a0+a1x+a2x2+a3x3+⋯
注意这里模型仍然是线性的，线性的意思是指模型中的斜率an没有互相进行乘法或除法操作。这里起作用的是我们将一维的x值投射到了更高的维度上，这样我们的线性模型就能拟合x和y之间更加复杂的联系。
"""

"""
多项式基本函数： Polynomial basis functions
这种多项式投射如此有用所以scikit_learn 内见了实现它的方法，就是ploynomialFeatures转换：
"""

# X = np.array([2, 3, 4])
# poly = PolynomialFeatures(3, include_bias=False)
# print(poly.fit_transform(X[:, None]))
# # 我们看到上例中使用这个转换器我们对每个值求幂将一维数组变成了三维数组，这个新的高维数据表示能应用到线性回归中
# # 正如我们在特征工程中看到的，实现这个任务的最优雅方法就是使用管道，这里我们创建一个7阶的多项式模型
# poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
# # 有了这样的转换方式，我们可以使用线性模型来拟合复杂得多的x和y的关系，例如下面带有噪音的正弦波：
# rng = np.random.RandomState(1)
# x = 10 * rng.rand(50)
# y = np.sin(x) + 0.1 * rng.randn(50)
# poly_model.fit(x[:, np.newaxis], y)
# print("x[:, np.newaxis]:", x[:, np.newaxis].shape)
# yfit = poly_model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.show()
"""
高斯基本函数： Gaussian basis functions
当然还有其他可用的基本函数，例如可以通过高斯函数叠加而不是多项式叠加来拟合模型，结果可能如下图所示：

"""


# class GaussianFeatures(BaseEstimator, TransformerMixin):
#     """
#     对一维数据进行均匀分布高斯转换
#     """
#     def __init__(self, N, width_factor=2.0):
#         self.N = N
#         self.width_factor = width_factor
#
#     @staticmethod
#     def _gauss_basis(x, y, width, axis=None):
#         arg = (x-y) / width
#         return np.exp(-0.5 * np.sum(arg**2, axis))
#
#     def fit(self, X, y = None):
#         # 沿着数据范围创建均匀分布的N个中心点
#         self.centers_ = np.linspace(X.min(), X.max(), self.N)
#         self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
#         return self
#
#     def transform(self, X):
#         return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)


# gauss_model = make_pipeline(GaussianFeatures(30), LinearRegression())
# gauss_model.fit(x[:, np.newaxis], y)
# yfit = gauss_model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.xlim(0, 10)
# plt.show()
"""
正则化： Regularization
将基本函数引入线性回归令我们的模型更加灵活，但是它很容易导致过拟合（参见超参数和模型验证中的讨论）。例如如果我们选择了太多的高斯函数，产生的结果就不太可靠了：
"""
# model = make_pipeline(GaussianFeatures(30),
#                       LinearRegression())
# model.fit(x[:, np.newaxis], y)
#
# plt.scatter(x, y)
# plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
#
# plt.xlim(0, 10)
# plt.ylim(-1.5, 1.5)
# plt.show()


"""
通过将数据投射到30维的空间上，该模型太过于灵活以至于当处于间隔距离较大的点之间的位置时候，会拟合成很极端的数据值，所以我们将高斯函数的系数也绘制在图表中，就可以看到原因：
"""
# def basis_plot(model, title=None):
#     fig, ax = plt.subplots(2, sharex=True)
#     model.fit(x[:, np.newaxis], y)
#     ax[0].scatter(x, y)
#     ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
#     ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
#     if title:
#         ax[0].set_title(title)
#     ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_)
#     ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0, 10))
#     plt.show()
#
#
# model = make_pipeline(GaussianFeatures(30), LinearRegression())
# basis_plot(model)
"""
下面的图展示了基本函数在每个位置的振幅，这时当使用基本函数叠加的点醒过拟合情况；邻近的基本函数的系数互相叠加到波峰和波谷。这种情况是错误的，
如果我们能在模型中限制这样的尖刺能解决这个问题，通过模型参数的大数值进行惩罚可以实现这个目标。这样的惩罚被称为正则，它有以下几种形式：
岭回归（L2正则化）: Ridge regression (L2 Regularization)
最常用的正则化方式被称为岭回归或L2正则化，有的时候也被叫做Tikhonov正则化。这个过程通过对模型系数的平方和（2-范数）进行乘法；在这个例子中，模型的乘法是
P=α∑n=1Nθ2n
其中α是控制乘法力度的参数。这类的乘法模型內建在Scikit-Learn中Ridge评估器中：
α参数是用来控制模型复杂度的关键开关。极限情况α→0时，恢复到标准线性回归结果；极限情况α→∞时，所有模型的响应都会被压缩。岭回归的一大优点是它能非常有效的计算，基本没有产生比原始线性回归模型更大的计算消耗。
"""
# model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
# basis_plot(model, title="Ridge Regression")

"""
Lasso 算法回归（L1 正则化）
另一个常用的正则化类型被称为lasso，通过惩罚回归系数绝对值和（1-范数）来实现：
P=α∑n=1N|θn|
虽然这在概念上非常类似岭回归，但是结果却大不相同：例如因为几何原因lasso回归更适合稀疏模型，即它倾向于将模型系数设置为0。
使用了lasso回归惩罚，大部分的系数都变成了0，也就是只有小部分的基本函数在模型中产生了作用。就像岭回归正则化，α参数调整惩罚的强度，这个参数应该通过比方说交叉验证（参见超参数和模型验证中的讨论）来确定。
"""
# from sklearn.linear_model import Lasso
# model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
# basis_plot(model, title='Lasso Regression')


"""示例"""
import pandas as pd

counts = pd.read_csv('data/Fremont_Bridge_Bicycle_Counter.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('data/BicycleWeather.csv', index_col='DATE', parse_dates=True)
print("counts :", counts.shape)
print("weather :", weather.shape)
daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']]  # 移除其他列
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)


def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """
    计算给定日期的日照时间
    axis 23.44 黄赤夹角
    latitude 47.61 西雅图纬度
    """
    # 2000年12月21日是冬至日，日照时间最短
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))  # np.radians： 角度转化成弧度
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.  # np.clip(a, a_min, a_max, out=None, **kwargs)  截取数组a中a_min和a_max范围的值，当小于a_min设置为a_min,当大于a_max设置为a_max


daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)
plt.show()

# 气温单位是0.1摄氏度，求平均值
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# 降雨量单位是0.1毫米，转换成英寸
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)
daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

# 让我们增加一列计数器从第一天开始计数，然后转换成经过的年的小数数值。该列会在每年进行循环：
daily['annual'] = (daily.index - daily.index[0]).days / 365.
print(daily.head())
# 有了数据后，我们可以选择使用哪些列来让线性回归模型进行拟合。我们设置fit_intercept=False，因为每天的数据都有着那一天自己的截距：
# 移除所有有空值的行
daily.dropna(axis=0, how='any', inplace=True)
# 用来拟合模型的列包括星期几、日照小时数、降水量、是否有雨、气温、该天的年计数
column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday',
                'daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual']
X = daily[column_names]
y = daily['Total']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily['predicted'] = model.predict(X)

daily[['Total', 'predicted']].plot(alpha=0.5)
plt.show()
