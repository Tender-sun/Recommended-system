#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

'''
矩阵分解的推荐算法
1.Main Idea
假设user和item有着共同现在的K个特征，组成矩阵P 和Q， 这两个矩阵是由原矩阵分解得到，由此可以得到原矩阵的估计

2.K大小的设定
K是超参数，没有直接的方式进行直接确定，只能通过交叉验证的方法来确定最优的K值，其大小表示user-item潜在的特征数，这些特征影响了user对item的评分。

注意：K的大小不能大于user或者item的数量。直观感受表示每个用户可以对应独立的特征导致过拟合。

3.正则化
主要作用是防止过拟合，添加额外参数beta
'''

class Matrix_Factorization(object):

    #k:超参数
    #beta:防止过拟合
    def __init__(self, K=10, alpha=0.01, beta=0.02, epoch=1000, regularization=True, random_state=100):

        self.R = None
        self.K = K
        self.P = None
        self.Q = None
        self.r_index = None
        self.r = None
        self.length = None
        self.alpha = alpha
        self.beta = beta
        self.epoch = epoch
        self.regularization = regularization
        self.random_state = random_state


    def fit(self, R):

        np.random.seed(self.random_state)
        self.R = R.values
        M, N = self.R.shape
        self.P = np.random.rand(M, self.K)
        self.Q = np.random.rand(N, self.K)

        self.r_index = self.R.nonzero()
        self.r = self.R[self.r_index[0], self.r_index[1]]
        self.length = len(self.r)


    def _comp_descent(self, index):

        r_i = self.r_index[0][index]
        r_j = self.r_index[1][index]

        p_i = self.P[r_i]
        q_j = self.Q[r_j]

        r_ij_hat = p_i.dot(q_j)
        e_ij = self.R[r_i, r_j] - r_ij_hat

        if self.regularization == True:
            descent_p_i = -2 * e_ij * q_j + self.beta * p_i
            descent_q_j = -2 * e_ij * p_i + self.beta * q_j
        else:
            descent_p_i = -2 * e_ij * q_j
            descent_q_j = -2 * e_ij * p_i

        return r_i, r_j, p_i, q_j, descent_p_i, descent_q_j


    def _update(self, p_i, q_j, descent_p_i, descent_q_j):

        p_i_new = p_i - self.alpha * descent_p_i
        q_j_new = q_j - self.alpha * descent_q_j

        return p_i_new, q_j_new


    def _estimate_r_hat(self):

        r_hat = self.P.dot(self.Q.T)[self.r_index[0], self.r_index[1]]

        return r_hat


    def start(self):

        epoch_num = 1
        while epoch_num <= self.epoch:
            for index in xrange(0, self.length):

                r_i, r_j, p_i, q_j, descent_p_i, descent_q_j = self._comp_descent(index)
                p_i_new, q_j_new = self._update(p_i, q_j, descent_p_i, descent_q_j)

                self.P[r_i] = p_i_new
                self.Q[r_j] = q_j_new

            r_hat = self._estimate_r_hat()
            e = r_hat - self.r
            error = e.dot(e)
            print 'The error is %s=================Epoch:%s' %(error, epoch_num)
            epoch_num += 1

        R_hat = self.P.dot(self.Q.T)
        return R_hat


if __name__ == '__main__':

    user_rating = pd.read_csv('./ml-latest-small/user-rating.csv', index_col=0)

    aa = Matrix_Factorization(K = 5)
    aa.fit(user_rating)
    aa.start()

    
