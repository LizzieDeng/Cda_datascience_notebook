#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：Cda_datascience_notebook -> 0504_FeatureEngineering
@IDE    ：PyCharm
@Author ：LizzieDeng
@Date   ：2021/1/27 21:41
@Desc   ：
=================================================="""
from sklearn.feature_extraction import  DictVectorizer

data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]
vec = DictVectorizer(sparse=False, dtype=int)
vec_trans = vec.fit_transform(data)
print(vec_trans)
vec_name = vec.get_feature_names()
print(vec_name)
vec = DictVectorizer(sparse=True, dtype=int)
print(vec.fit_transform(data))

sample = ['problem of evil',
          'evil queen',
          'horizon problem']
from sklearn.feature_extraction.text import CountVectorizer
vec_sample = CountVectorizer()
X = vec_sample.fit_transform(sample)
print("X is {}".format(X))
import pandas as pd
pd_data = pd.DataFrame(X.toarray(), columns=vec_sample.get_feature_names())
print('pd_data is \n{}'.format(pd_data))
from sklearn.feature_extraction.text import TfidfVectorizer
vec_tfid = TfidfVectorizer()
X_tfid = vec_tfid.fit_transform(sample)
pd_tfid = pd.DataFrame(X_tfid.toarray(), columns=vec_tfid.get_feature_names())
print('pd_tfid is \n{}'.format(pd_tfid))

# 


