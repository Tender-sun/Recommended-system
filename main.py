#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from data import *
from measure import *
from Content_based_recommendation import *
from Collaborative_Filtering_recommendation import *
from Matrix_Factorization import *


if __name__ == '__main__':

    '''
    #1.处理数据
    #生成以下两个CSV文件，仅执行一次
    movies.csv-》movies_feature.csv：创建电影的年份特征
    rating.csv-》user-rating.csv:创建user-item评分矩阵
    '''
    #data_preprocess()
    

    #2.导入数据(注:根目录格式为'./ml-..../...')
    movies_feature = pd.read_csv('./ml-latest-small/movies_feature.csv', index_col=0)
    user_rating = pd.read_csv('./ml-latest-small/user-rating.csv', index_col=0)
    print user_rating
    
    
    #3.将原始user-item评分矩阵分为训练集和测试集
    train, test = train_test_split(user_rating)


    '''
    #4和5生成两个csv文件，仅执行一次
    
    #4.使用协同过滤算法来估计评分
    count = 0
    total = float(train.shape[0])
    
    for idx, user in train.iterrows():
        unrated_index = user[user == 0].index.values
        unrated_index_ = map(int, unrated_index)        
        rates_lst = CF_recommend_estimate(train, idx, unrated_index_, 50)#para:训练集+用户(关键)+物品区间+topK

        train.loc[idx, unrated_index] = rates_lst#选取相应行数据

        #提示进度
        count += 1
        if count % 100 == 0:
            presentage = round((count / total) * 100)
            print 'Completed %d' % presentage + '%'
    
    train.to_csv('./ml-latest-small/pred_ratings_CF.csv')
    
    #注:这两个csv文件的生成时间较长，电脑负荷较大，耐心。
    #其中，两个文件大小均为105M
    #生成数据，进一步评价，计算MSE和RMSE，这里还是只用MSE(区别不大)

    
    #5.使用基于内容的推荐算法来估计评分
    count = 0
    total = float(train.shape[0])
    
    for idx, user in train.iterrows():        
        unrated_index = user[user == 0].index.values
        rates_lst = []
    
        for item in unrated_index:            
            rate_h = CB_recommend_estimate(user, movies_feature, int(item))       
            rates_lst.append(rate_h)
           
        train.loc[idx, unrated_index] = rates_lst#选取相应行数据
     
        #提示进度
        count += 1
        if count % 100 == 0:
            presentage = round((count / total) * 100)
            print 'Completed %d' % presentage + '%'
            
    train.to_csv('./ml-latest-small/pred_ratings_CB.csv')
    '''

    
    #6.评估：CF和CB的MSE和RMSE
    pred_CB = pd.read_csv('./ml-latest-small/pred_ratings_CB.csv', index_col=0)
    pred_CF = pd.read_csv('./ml-latest-small/pred_ratings_CF.csv', index_col=0)
    nonzero_index = user_rating.values.nonzero()
    #print pred_CB
    #print pred_CF
    #print nonzero_index
    
    actual = user_rating.values[nonzero_index[0], nonzero_index[1]]
    pred_CB = pred_CB.values[nonzero_index[0], nonzero_index[1]]
    pred_CF = pred_CF.values[nonzero_index[0], nonzero_index[1]]
    print actual
    print pred_CB
    print pred_CF
    
    print 'MSE of CB is %s' % comp_mse(pred_CB, actual)
    print 'RMSE of CB is %s' % comp_rmse(pred_CB, actual)
    
    print 'MSE of CF is %s' % comp_mse(pred_CF, actual)
    print 'RMSE of CF is %s' % comp_rmse(pred_CF, actual)

  
    #7.评估：MF的MSE和RMSE   
    MF_estimate = Matrix_Factorization(K=10, epoch=10)
    MF_estimate.fit(train)
    R_hat = MF_estimate.start()
    non_index = test.values.nonzero()
    
    pred_MF = R_hat[non_index[0], non_index[1]]
    actual = test.values[non_index[0], non_index[1]]
    
    print 'MSE of MF is %s' % comp_mse(pred_MF, actual)
    print 'RMSE of MF is %s' % comp_rmse(pred_MF, actual)