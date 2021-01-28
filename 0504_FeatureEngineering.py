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