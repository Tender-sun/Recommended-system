#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

'''
基于内容的推荐算法（Content-based recommendation）

1.Main Idea
对于用户C，推荐与用户C之前感兴趣的items相似度高的items。

2.建立item画像(Item profiles)
创建关于item的特征描述

e.g. moive: author, title, actor, director....

​	text: setof important words in doccument....

3.建立User画像(User profiles)
创建关于User的的特征描述，使用与item相同的特征，每个特征值与user的行为有关

4.相似度度量
夹角余弦

sim(u,c) = cos<u, c>

5.局限性
难以确定合适的特征

不能推荐用户兴趣以外的内容，且用户要有较为广泛的兴趣

e.g. 有一样东西大家都说好，但某用户之前并无此类兴趣则不能推荐给该用户

无法对新用户进行推荐
'''


def cos_measure(item_feature_vector, user_rated_items_matrix):
    """
    计算item之间的余弦夹角相似度
    :param item_feature_vector: 待测量的item特征向量
    :param user_rated_items_matrix: 用户已评分的items的特征矩阵
    :return: 待计算item与用户已评分的items的余弦夹角相识度的向量
    """
    x_c = (item_feature_vector * user_rated_items_matrix.T) + 0.0000001
    mod_x = np.sqrt(item_feature_vector * item_feature_vector.T)
    mod_c = np.sqrt((user_rated_items_matrix * user_rated_items_matrix.T).diagonal())
    cos_xc = x_c / (mod_x * mod_c)

    return cos_xc


def estimate_rate(user_rated_vector, similarity):
    """
    估计用户对item的评分
    :param user_rated_vector: 用户已有item评分向量
    :param similarity: 待估计item和已评分item的相识度向量
    :return:用户对item的评分的估计
    """
    rate_hat = (user_rated_vector * similarity.T) / similarity.sum()
    return rate_hat[0, 0]


def comp_user_feature(user_rated_vector, item_feature_matrix):
    """
    根据user的评分来计算得到user的喜好特征
    :param user_rated_vector: user的评分向量
    :param item_feature_matrix: item的特征矩阵
    :return: user的喜好特征
    """
    #user评分的均值
    user_rating_mean = user_rated_vector.mean()
    # 分别得到user喜欢和不喜欢item的向量以及item对应的引索(以该user的评分均值来划分)
    user_like_item = user_rated_vector.loc[user_rated_vector >= user_rating_mean]
    user_unlike_item = user_rated_vector.loc[user_rated_vector < user_rating_mean]

    user_like_item_index = map(int, user_like_item.index.values)
    user_unlike_item_index = map(int, user_unlike_item.index.values)

    user_like_item_rating = np.matrix(user_like_item.values)
    user_unlike_item_rating = np.matrix(user_unlike_item.values)

    #得到user喜欢和不喜欢item的特征矩阵
    user_like_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_like_item_index, :].values)
    user_unlike_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_unlike_item_index, :].values)

    # print user_like_item, user_unlike_item

    #计算user的喜好特征向量，以其对item的评分作为权重
    weight_of_like = user_like_item_rating / user_like_item_rating.sum()
    weight_of_unlike = user_unlike_item_rating / user_unlike_item_rating.sum()

    #计算user的喜欢特征和不喜欢特征以及总特征
    user_like_feature = weight_of_like * user_like_item_feature_matrix
    user_unlike_feature = weight_of_unlike * user_unlike_item_feature_matrix
    user_feature_tol = user_like_feature - user_unlike_feature

    return user_feature_tol


def CB_recommend_estimate(user_feature, item_feature_matrix, item):
    """
    基于内容的推荐算法对item的评分进行估计
    :param item_feature_matrix: 包含所有item的特征矩阵
    :param user_feature: 待估计用户的对item的评分向量
    :param item: 待估计item的编号
    :return: 基于内容的推荐算法对item的评分进行估计
    """
    # #得到item的引索以及特征矩阵
    # item_index = item_feature_matrix.index
    # item_feature = item_feature_matrix.values

    #得到所有user评分的item的引索
    user_item_index = user_feature.index

    #某一用户已有评分item的评分向量和引索以及item的评分矩阵
    user_rated_vector = np.matrix(user_feature.loc[user_feature > 0].values)
    user_rated_items = map(int, user_item_index[user_feature > 0].values)
    user_rated_items_matrix = np.matrix(item_feature_matrix.loc[user_rated_items, :].values)


    #待评分itme的特征向量，函数中给出的是该item的Id
    item_feature_vector = np.matrix(item_feature_matrix.loc[item].values)

    #得到待计算item与用户已评分的items的余弦夹角相识度的向量
    cos_xc = cos_measure(item_feature_vector, user_rated_items_matrix)

    #计算uesr对该item的评分估计
    rate_hat = estimate_rate(user_rated_vector, cos_xc)

    return rate_hat

#推荐功能
def CB_recommend_top_K(user_feature, item_feature_matrix, K):
    """
    计算得到与user最相似的Top K个item推荐给user
    :param user_feature: 待推荐用户的对item的评分向量
    :param item_feature_matrix: 包含所有item的特征矩阵
    :param K: 推荐给user的item数量
    :return: 与user最相似的Top K个item的编号
    """

    # 得到user已评分和未评分的item向量
    user_rated_vector = user_feature.loc[user_feature > 0]
    user_unrated_vector = user_feature.loc[user_feature == 0]

    #未评分item的特征矩阵
    user_unrated_item_index = map(int, user_unrated_vector.index.values)
    user_unrated_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_unrated_item_index, :].values)

    #user喜好总特征
    user_feature_tol = comp_user_feature(user_rated_vector, item_feature_matrix)
    #得到相似度并进行排序
    similarity = list(np.array(cos_measure(user_feature_tol, user_unrated_item_feature_matrix))[0])
    key = {'item_index': user_unrated_item_index,
           'similarity': similarity}
    item_sim_df = pd.DataFrame(key)
    item_sim_df.sort_values('similarity', ascending=False, inplace=True)

    return item_sim_df.iloc[:K, 0].values


if __name__ == '__main__':
    
    movies_feature = pd.read_csv('./ml-latest-small/movies_feature.csv', index_col=0)
    user_rating = pd.read_csv('./ml-latest-small/user-rating.csv', index_col=0)
    user_feature = user_rating.loc[100,:]
    
    print CB_recommend_estimate(user_feature, movies_feature, 19)#user特征+物品特征+待估分item编号(关键)