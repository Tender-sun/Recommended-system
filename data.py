#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re

def _judge_of_year(year):
    """
    判断电影上映年份所属的年代
    :param year: int, 电影上映年代
    :return: str, 所属年代的字符串
    """
    if isinstance(year, int):

        if year < 2000:
            return '90s'#90年代及以前
        elif year < 2010:
            return '00s'#2000-2010
        else:
            return '10s'#2010后
    else:
        raise ValueError('Type error!')#输入不合法


def train_test_split(user_rating, random_state=100):
    """
    将原始user评分矩阵划分为训练集和验证集
    :param user_rating: 原始user评分矩阵
    :return: user评分训练集和验证集
    """
    np.random.seed(random_state)
    ratings_index = user_rating.index
    ratings_col = user_rating.columns
    ratings = user_rating.values
    print ratings
    test = np.zeros(user_rating.shape)
    train = ratings.copy()
    for user in xrange(user_rating.shape[0]):
        #np.random.choice(a, size=None, replace=True, p=None)
        #a:一维数组或整数
        #size:生成样本的大小
        #replace:bool类型  False表示样本中不允许有重复值 True......
        #p:给定数组中元素出现的概率
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    assert (np.all((train * test) == 0))
    train = pd.DataFrame(train,
                         columns=ratings_col,
                         index= ratings_index)
    test = pd.DataFrame(test,
                        columns=ratings_col,
                        index=ratings_index)
    return train, test#返回处理好的训练集和测试集


def data_preprocess():
    #创建user-item评分矩阵
    print "Loading data..."
    rating_data = pd.read_csv('./ml-latest-small/ratings.csv')
    user_id = rating_data['userId'].unique()
    movie_id = rating_data['movieId'].unique()
    rating_matrix = np.zeros([len(user_id),len(movie_id)])
    rating_matrix = pd.DataFrame(rating_matrix, index=user_id, columns=movie_id)

    print "Converting data..."
    count = 0
    user_num= len(user_id)
    for uid in user_id:
        user_rating = rating_data[rating_data['userId'] == uid].drop(['userId', 'timestamp'], axis=1)
        user_rated_num = len(user_rating)

        for row in xrange(0, user_rated_num):
            movieId = user_rating['movieId'].iloc[row]
            rating_matrix.loc[uid, movieId] = user_rating['rating'].iloc[row]

        count += 1
        if count % 100 == 0:
            completed_percentage = round(float(count) / user_num * 100)
            print "Completed %s" % completed_percentage + "%"

    rating_matrix.to_csv('./ml-latest-small/user-rating.csv')


    #创建电影的年份特征
    print "Loading data..."
    movies_data = pd.read_csv('./ml-latest-small/movies.csv')
    movies_num = len(movies_data)

    print "Creating the feature of years/n"
    years = ["Unknowyear", "90s", "00s", "10s"]  #定义年份的划分区间
    for year in years:
        movies_data[year] = np.zeros([movies_num, 1])

    count = 0
    for i in xrange(0, movies_num):
        title = movies_data.loc[i]['title']

        pattern = re.compile(r'\d{4}')  # 查找电影年份
        result = pattern.findall(title)

        #判断电影所属年份区间
        if len(result) != 0:
            year_of_movie = result[-1]
            year_col = _judge_of_year(int(year_of_movie))
            movies_data.loc[i, year_col] = 1
        else:
            movies_data.loc[i, 'Unknowyear'] = 1

        count += 1
        if count % 1000 == 0:
            completed_percentage = round(float(count) / movies_num * 100)
            print "Completed %s" % completed_percentage + "%"


    #创建电影的分类特征
    print "Creating the feature of genres/n"
    #得到该电影数据集中分类的类别列表
    genres_list = set()
    for genre in movies_data['genres']:
        genres_list = genres_list | set(genre.split('|'))

    #给每个类别创建一个维度加到原电影数据集中
    genres_list = list(genres_list)
    for genre in genres_list:
        movies_data[genre] = np.zeros([movies_num, 1])


    count = 0
    for j in xrange(0, movies_num):
        movie_genres = movies_data.loc[j, 'genres'].split('|')
        for k in movie_genres:
            movies_data.loc[j, k] = 1

        count += 1
        if count % 1000 == 0:
            completed_percentage = round(float(count) / movies_num * 100)
            print "Completed %s" % completed_percentage + "%"

    movies_data.set_index('movieId', inplace=True)
    movies_data.drop(['title', 'genres'], axis=1, inplace=True)
    movies_data.to_csv('./ml-latest-small/movies_feature.csv')


if __name__ == '__main__':
    data_preprocess()
    
