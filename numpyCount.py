import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from datetime import datetime
from dateutil import parser
import time
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessClassifier
from pandas_datareader import data
import numexpr as ne
seaborn.set()


def numpy_learner_func():
    # numpy 计数
    rng = np.random.RandomState(0)
    x_data = rng.randint(10, size=(3, 4))
    print("rng:{}".format(rng))
    print("x_data{}".format(x_data))
    # 计小于6的数 , np.count_nonzero, np.sum, np.where
    num1 = np.count_nonzero(x_data < 6)
    num2 = np.sum(x_data < 6)
    num3 = np.any(x_data < 6)
    num4 = np.all(x_data < 6)
    num5 = np.where(x_data < 6)[0]
    print(x_data < 6, num3, num4, num5, num5.shape[0])
    print("num1 is {}".format(num1))
    print("num2 is {}".format(num2))
    print(x_data[x_data < 6])
    print(9 and 0)

    # numpy newaxis 给数组新增维度
    x = np.arange(3)
    print(x, x.shape)
    x1 = x[:, np.newaxis]
    print(x1, x1.shape)
    x2 = x[:, np.newaxis, np.newaxis]
    print(x2, x2.shape)

    x3 = np.zeros(10)
    np.add.at(x3, [0, 1, 5], 1)
    print(x3)
    # print("x4 is {}".format(x4))
    i = [2, 3, 3, 4, 4, 4]
    x3[i] += 1
    print(x3)
    # np.random.seed(42)
    x_np = np.random.randn(100)
    bins = np.linspace(-5, 5, 20)
    # zeros_like 返回与参数一样shape的数组
    counts = np.zeros_like(bins)
    print("counts is {}".format(counts))
    # np.searchsorted 将数字x_np插入到排好序的list中，返回相应的下标
    j = np.searchsorted(bins, x_np)
    print("j is {}".format(j))
    # np.searchsorted()

    # ## numpy 排序  np.sort()返回排好序的新数组
    srt_array = np.array([2, 1, 4, 3, 5])
    print("sorted:{}".format(np.sort(srt_array)))
    # x.sort() Python内置函数sort(),对原数组进行排序，返回原数组
    print("x.sort() is {}".format(srt_array.sort()))
    sorted_arr = np.array([99, 0, 3, 1, 90])
    # np.argsort()返回数组中排序之后的下标
    print("np.argsort(srt_array) is {}".format(np.argsort(sorted_arr)))

    # np.sort(axis = None)按照维度排序
    axis_arr = np.random.RandomState(42).randint(0, 10, (4, 6))
    print("the array is {}".format(axis_arr))
    print("sort each column of axis_arr, returns {}".format(np.sort(axis_arr, axis=0)))
    print("sort each row of axis_arr, returns {}".format(np.sort(axis_arr, axis=1)))
    # 部分排序， 分区排序
    np_part = np.array([3, 8, 4, 99, 5, 1, 88])  # 1 3 4 5 88 99      3,4, 1, 5,8, 99,  88
    print("np_part partition sorted is {}".format(np.partition(np_part, 3,)))


def K_nearest_neighbors_func():
    X = np.random.RandomState(42).rand(10, 2)   # 10X2 array
    plt.scatter(X[:, 0], X[:, 1], s=100)
    x_newaxis = X[:, np.newaxis, :]
    print("X[:, np.newaxis, :]:", x_newaxis)
    print(x_newaxis.shape)
    x_newaxis_1 = X[np.newaxis, :, :]
    print("x_newaxis_1:", x_newaxis_1)
    print(x_newaxis_1.shape)
    diff_newaxis = x_newaxis - x_newaxis_1
    print("diff_newaxis:", diff_newaxis,  diff_newaxis.shape)
    sq_differences = diff_newaxis ** 2
    dist_sq = sq_differences.sum(-1)    # axis 从倒数第2个到第一个
    print("dist_sq:", dist_sq, sq_differences.shape, dist_sq.shape)
    eye_dist_sq = dist_sq.diagonal()  # 返回指定矩阵的对角线
    print("eye_dist_sq is {}".format(eye_dist_sq))
    nearest = np.argsort(dist_sq, axis=1)   # 对列进行从小到大排序，返回排好序之后的索引值
    K = 2
    nearest_partition = np.argpartition(dist_sq, K+1, axis=1)  # 分区排序，返回排好序的索引值
    # print("nearest_partition.shape is {}".format(nearest_partition.shape))
    # #
    # # dis_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=-1)
    for i in range(X.shape[0]):
        for j in nearest_partition[i, :K+1]:
            plt.plot(*zip(X[j], X[i]), color='black')
    # k_nearest_neighbors_loop_func(X, K)
    plt.show()


def k_nearest_neighbors_loop_func(X, K):
    all_dist = {}
    index_dict = {}
    # 计算每个点与其他点之间的距离并按序排列
    for i in range(X.shape[0]):
        start_point = X[i, :]
        start_point_dis = {}
        for j in range(X.shape[0]):
            if i != j:
                dis = np.sqrt((start_point[0] - X[j, 0])**2 + (start_point[1] - X[j, 1])**2)
                # start_point_dis.append(dis)
                start_point_dis[j] = dis
        # 字典排序,按照值
        sorted_start_point_dis = {}
        # for item in dict_a.items():
        #     print(item)
        #     out.append((item[1], item[0]))
        # print(out, sorted(out))
        inter_list = sorted(start_point_dis.items(), key = lambda kv:(kv[1], kv[0]))
        for each in inter_list:
            sorted_start_point_dis[each[0]] = each[1]
        all_dist[i] = list(sorted_start_point_dis.keys())[:K]
        # 取出最近的两个点index
    for a in range(X.shape[0]):
        for b in all_dist[a]:
            print("a, b", a, b)
            plt.plot(*zip(X[a, :], X[b, :]), color='blue')

    plt.show()
    # print(all_dist)


def pandas_learner():
    # pandas 里面的index 是不可变数组或者允许存在重复值的有序集合
    indA = pd.Index([1, 3, 5, 7, 9])
    indB = pd.Index([2, 3, 5, 7, 11])
    index1 = indA & indB   # 交集
    index2 = indA | indB   # 全集
    index3 = indA ^ indB   # 差集
    print(index1, index2, index3)

    data = pd.Series([0.25, 0.5, 0.75, 1.0],
                     index=['a', 'b', 'c', 'd'])
    print(data['b'])
    print('a' in data)
    print(data.keys())
    print(list(data.items()))
    data['e'] = 1.25
    print(data['a': 'c'])  # 切片， 包含c列
    print(data[0:2])
    print(data[(data > 0.3) & (data < 0.8)])
    print(data[['a', 'e']])
    # loc 根据列标签索引访问
    print(data[1])
    print(data[1:3])
    print(data.loc['a'])
    # iloc根据行下标访问行
    print(data.iloc[1])
    print(data.iloc[1:3])


def pandas_null():
    valsl = np.array([1, np.nan, 3, 4])
    print(valsl.dtype)
    print(1+np.nan)
    print(0*np.nan)
    print(np.sum(valsl), np.min(valsl), np.max(valsl))  # 任何累加和计算，最大值，最小值聚类函数中含有nan，其结果都是nan
    print(np.nansum(valsl), np.nanmin(valsl), np.nanmax(valsl))  # 忽略nan值，计算累加和，最小值，最大值
    print(np.nan == None)
    data = pd.Series([1, np.nan, 'hello', None])
    print(data.isnull())
    print(data.notnull())
    print(data[data.notnull()])
    print("dropnan:", data.dropna())
    data_df = pd.DataFrame([[1, np.nan, 2], [2, 3, 5], [np.nan, 4, 6]])
    print(data_df.dropna())
    print(data_df.dropna(axis='columns'))
    data_df[3] = np.nan
    print(data_df.dropna(axis='columns', how='all'))
    print(data_df.dropna(axis='columns', how='any'))
    print(data_df.dropna(axis='rows', thresh=3))


def numpy_learner():
    df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data': range(6)}, columns=['key', 'data'])
    print('df is {} \n'.format(df))
    df.groupby('key')
    print("df.groupby('key')".format(df))
    print('DataFrames is'.format(df.groupby('key').sum()))
    print('sum is sum() is {}'.format(df.groupby('key').sum()))
    print("planets.groupby('method')".format())


def pandas_aggregation_group():
   rng = np.random.RandomState(42)
   ser = pd.Series(rng.rand(5))
   print(ser.mean())
   print(ser.sum())
   df = pd.DataFrame({'A': rng.rand(5), 'B': rng.rand(5)})
   print(df.mean(axis='columns'))
   df_data = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'data': range(6)}, columns=['key', 'data'])
   print(df_data)
   # print()


def learn_pivot_table():
    # openurl失败，将数据集下载到本地，并引用
    titanic = seaborn.load_dataset('titanic', cache=True, data_home="./seaborn-data")
    print(titanic.head())
    print(titanic.groupby('sex')[['survived']].mean())
    print(titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack())
    # 透视表
    t_pivot_table = titanic.pivot_table('survived', index='sex', columns='class')
    print(t_pivot_table)
    # fare = pd.qcut(titanic, 2)
    age = pd.cut(titanic['age'], [0, 18, 80])
    age_table = titanic.pivot_table('survived', ['sex', age], 'class')
    print(age_table)
    fare = pd.qcut(titanic['fare'], 2)
    fare_table = titanic.pivot_table('survived', ['sex', age], [fare, 'class'])
    print(fare_table)


def working_with_strings():
    # pandas 能够向量化处理string类型的数据
    data = ['Peter', 'Paul', None, 'MARY', 'gUIDO']
    names = pd.Series(data)
    print(names)
    print(names.str.capitalize())
    monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam', 'Eric Idle', 'Terry Jones', 'Michael Palin'])
    print(monte.str.lower())
    print(monte.str.len())
    print(monte.str.startswith('T'))
    print(monte.str.split())
    # 使用正则表达式
    print(monte.str.extract('([A-Za-z]+)', expand=False))
    print(monte.str.findall(r'^[^AEIOU].*[^aeiou]$'))
    print(monte.str[0:3])
    print(monte.str.split().str.get(-1))
    full_monte = pd.DataFrame({'name': monte, 'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C', 'B|C|D']})
    print(full_monte)
    s = pd.Series(list('abcd'))
    print('s is {}'.format(s))
    print(pd.get_dummies(s))   # get_dummies() 将类别变量变成指标变量
    print(full_monte['info'].str.get_dummies('|'))


def working_with_time_series():
    datetime_1 = datetime(year=2020, month=7, day=4)
    print(datetime_1)
    date = parser.parse('4th of july, 2020')
    print(date)
    print(date.strftime('%A'))  # 返回时期，返回星期几
    print(date.strftime('%a'))  # 返回时期，返回星期几缩写
    print(date+pd.to_timedelta(np.arange(12), 'D'))
    # print()
    index = pd.DatetimeIndex(['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'])
    data_index = pd.Series([0, 1, 2, 3], index=index)
    print('data is {}'.format(data_index))
    print(data_index['2014-07-04':'2015-07-04'])
    print(data_index['2015'])
    dates = pd.to_datetime([datetime(2015, 7, 3), '4th of july, 2015', '2015-Jul-6', '07-07-2015', '20150708'])
    print('dates is {}'.format(dates))
    print('subtracted of timeIndex is timedeltaIndex {}'.format(dates-dates[0]))
    print('to_period is {}'.format(dates.to_period('D')))
    # 日期序列, timestamps: pd.period_range()
    # period: pd.period_range()
    # time delta: pd.timedelta_range()
    print(pd.date_range('2015-07-03', '2015-07-10'))
    print(pd.date_range('2015-07-03', periods=8))  # period控制时间长度范围，默认单位为天
    print(pd.date_range('2015-07-03', periods=8, freq='H'))
    print(pd.date_range('2015-07', periods=8, freq='M'))
    print(pd.timedelta_range(0, periods=10, freq='H'))
    goog = data.DataReader('GOOG', start='2004', end='2016', data_source='yahoo')
    print(goog.head())
    goog = goog['Close']
    # print(goog)
    goog.plot()
    plt.show()
    goog.resample('BA').mean().plot(style=':')
    goog.asfreq('BA').plot(style='--')
    plt.legend(['input', 'resample', 'asfreq'], loc='upper left')
    fig, ax = plt.subplots(2, sharex=True)
    data = goog.iloc[:10]
    data.asfreq('D').plot(ax=ax[0], marker='o')
    data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
    data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
    ax[1].legend(['back-fill', 'forward-fill'])
    goog.plot(ax=ax[0])
    goog.shift(900).plot(ax=ax[1])
    goog.shift(900).plot(ax=ax[2])
    local_max = pd.to_datetime('2007-11-05')
    offset = pd.Timedelta(900, 'D')
    ax[0].legend(['input'], loc=2)
    ax[0].get_xticktables()[2].set(weight='heavy', color='red')
    ax[0].axvline(local_max, alpha=0.3, color='red')

    ax[1].legend(['shift(900)'], loc=2)
    ax[1].get_xticklabels()[2].set(weight='heavy', color='red')
    ax[1].axvline(local_max+offset, alpha=0.3, color='red')

    ax[2].legend(['tshift(900)'], loc=2)


def performance_eval_and_query():
    # large data numexpr 比numpy更快，因为numpy会产生中间变量，占内存
    a = np.random.rand(10000000)
    b = np.random.rand(10000000)
    t1 = time.time()
    c = 2 * a + 3 * b
    print("tt1 is {}".format(time.time() - t1))
    t2 = time.time()
    d = ne.evaluate('2 * a + 3 * b')
    print("tt2 is {}".format(time.time() - t2))

    # pandas 中的eval()对于大量数据来说，计算快速
    nrows, nclos = 100000, 100
    rng = np.random.RandomState(42)
    df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, nclos)) for i in range(4))
    t3 = time.time()
    pandas_1 = df1 + df2 + df3 + df4
    print("pandas add is {}".format(time.time() - t3))
    t4 = time.time()
    pandas_2 = pd.eval('df1 + df2 + df3 + df4')
    print("pandas eval add is {}".format(time.time() - t4))
    # dataframe.eval()对列名操作
    dddf = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
    print(dddf.head())
    res1 = (dddf['A'] + dddf['B']) / (dddf['C']-1)
    res2 = pd.eval("(dddf.A + dddf.B) / (dddf.C-1)")
    print('np.allclose is {}'.format(np.allclose(res1, res2)))
    res3 = dddf.eval('(A+B)/(C-1)')
    print('np.allclose(res1, res3) is {}'.format(np.allclose(res1, res3)))
    dddf.eval('D=(A+B)/C', inplace=True)
    print('eval to create new columns {}'.format(dddf.head()))
    dddf.eval('D=(A-B)/C', inplace=True)
    print("eval to modify existing column {}".format(dddf.head()))
    # @ 表示变量名不是列名，@只能用于Dataframe.eval()不能用于pandas.eval()
    column_mean = dddf.mean(1)
    ress_1 = dddf['A'] + column_mean
    ress_2 = dddf.eval('A + @column_mean')
    print('np.allclose(ress_1,ress_2) {}'.format(np.allclose(ress_1, ress_2)))
    # ## dataframe.eval()不能使用比较，这是需要用到Dataframe.query(), pd.eval()可以使用比较语句
    compare_1 = dddf[(dddf.A < 0.5) & (dddf.B < 0.5)]
    compare_2 = pd.eval('dddf[(dddf.A < 0.5) & (dddf.B < 0.5)]')
    compare_3 = dddf.query('A < 0.5 and B < 0.5')
    print('compare in pd.eval {}'.format(np.allclose(compare_1, compare_2)))
    print('compare in dataframe.query() {}'.format(np.allclose(compare_1, compare_3)))


def simple_line_plot_04_01():
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 10, 1000)
    # line color
    plt.plot(x, np.sin(x-0), color='blue', linestyle='solid')      # '-'
    plt.plot(x, np.cos(x-1), color='g', linestyle='dashed')        # '--'
    plt.plot(x, np.cos(x-2), color='0.75', linestyle='dashdot')    # '-.'
    plt.plot(x, np.cos(x-3), color='#FFDD44', linestyle='dotted')  # ':'
    plt.plot(x, np.cos(x-4), color=(1.0, 0.2, 0.3))
    plt.plot(x, np.cos(x-5), color='chartreuse')


def Errorbars_04_03():
    plt.style.use('seaborn-whitegrid')
    x = np.linspace(0, 10, 50)
    dy = 0.8
    y = np.sin(x) + dy * np.random.randn(50)
    plt.errorbar(x, y, yerr=dy, fmt='.k')
    # plt.plot(x, y)
    plt.plot(x, [y1+0.8 for y1 in y])
    plt.plot(x, [y1-0.8 for y1 in y])
    plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)


def histograms_and_binnings():
    plt.style.use('seaborn-white')
    data = np.random.randn(1000)
    # plt.hist(data)
    # plt.hist(data, bins=30, density=True, alpha=0.3,
    #          histtype='stepfilled', color='steelblue',
    #          edgecolor='none')
    # plt.hist(data, bins=30, density=True, alpha=0.3, histtype='stepfilled', color='steelblue', edgecolor='none')
    # plt.show()
    x1 = np.random.normal(0, 0.8, 1000)
    x2 = np.random.normal(-2, 1, 1000)
    x3 = np.random.normal(3, 2, 1000)
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    plt.hist(x1, **kwargs)
    plt.hist(x2, **kwargs)
    plt.hist(x3, **kwargs)

    plt.show()
    mean = [0, 0]
    cov = [[1, 1], [1, 2]]
    x, y = np.random.multivariate_normal(mean, cov, 10000).T
    plt.hist2d(x, y, bins=30, cmap='Blues')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    data = np.vstack([x, y])
    kde = gaussian_kde(data)
    xgrid = np.linspace(-3.5, 3.5, 40)
    ygrid = np.linspace(-6, 6, 40)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto', extent=[-3.5, 3.5, -6, 6], cmap='Blues')
    cb1 = plt.colorbar()
    cb1.set_label('density')


def plot_contour():
    x = np.arange(1, 10)
    y = x.reshape(-1, 1)
    h = x * y

    cs = plt.contourf(h, levels=[10, 30, 50],
                      colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    plt.show()



if __name__ == "__main__":
    plot_contour()
    # pandas_aggregation_group()
    # K_nearest_neighbors_func()
    # learn_pivot_table()
    # pandas_learner()
    # pandas_null()
    # numpy_learner()
    # working_with_strings()
    # working_with_time_series()
    # performance_eval_and_query()
    # simple_line_plot_04_01()
    # Errorbars_04_03()
    # histograms_and_binnings()
