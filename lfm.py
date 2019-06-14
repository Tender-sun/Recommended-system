# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import heapq
import operator

import numpy

#建立隐语义模型，并使用随机梯度下降优化
def factorization(train, bias=True, svd=True, svd_pp=False, steps=25, gamma=0.04, slow_rate=0.93, Lambda=0.1, k=15,
                  ratio=None, seed=0):
    """
    :param train: 训练集
    :param bias: 是否计算偏移
    :param svd: 是否使用奇异值分解
    :param steps: 迭代次数
    :param gamma: 步长
    :param slow_rate: 步长减缓的系数
    :param Lambda: 正则化参数
    :param k: 奇异值分解向量长度
    """
    global _user_items
    _user_items = train
    numpy.random.seed(seed)
    global _bias, _svd, _svd_pp, _k
    _bias = bias
    _svd = svd
    _svd_pp = svd_pp
    _k = k
    global _bu, _bi, _pu, _qi, _z, _movie_list, _movie_set, _avr, _tot
    _bu = {}
    _bi = {}
    _pu = {}
    _qi = {}
    _z = {}
    sqrt_item_len = {}
    _movie_list = []
    _avr = 0
    _tot = 0
    y = {}
    for user, items in _user_items.iteritems():
        if _bias:
            _bu.setdefault(user, 0)
        if _svd:
            _pu.setdefault(user, numpy.random.random((_k, 1)) / numpy.sqrt(_k))
        if _svd_pp:
            sqrt_item_len.setdefault(user, numpy.sqrt(len(items)))
        for item, rating in items.iteritems():
            _movie_list.append(item)
            if _bias:
                _bi.setdefault(item, 0)
            if _svd:
                _qi.setdefault(item, numpy.random.random((_k, 1)) / numpy.sqrt(_k))
            if _svd_pp:
                y.setdefault(item, numpy.zeros((_k, 1)))
            _avr += rating
            _tot += 1
    _movie_set = set(_movie_list)
    _avr /= _tot
    for step in xrange(steps):
        rmse_sum = 0
        mae_sum = 0
        for user, items in _user_items.iteritems():
            samples = items if not ratio else __random_negative_sample(items, ratio)
            s = numpy.zeros((_k, 1))
            if _svd_pp:
                _z[user] = numpy.zeros((_k, 1))
                for item, rating in samples.iteritems():
                    _z[user] += y[item] / sqrt_item_len[user]
            for item, rating in samples.iteritems():
                eui = rating - __predict(user, item)
                rmse_sum += eui ** 2
                mae_sum += abs(eui)
                if _bias:
                    _bu[user] += gamma * (eui - Lambda * _bu[user])
                    _bi[item] += gamma * (eui - Lambda * _bi[item])
                if _svd_pp:
                    s += eui * _qi[item] / sqrt_item_len[user]
                if _svd:
                    _pu[user], _qi[item] = _pu[user] + gamma * (eui * _qi[item] - Lambda * _pu[user]), _qi[
                        item] + gamma * (eui * (_pu[user] + _z[user] if _svd_pp else _pu[user]) - Lambda * _qi[item])
            if _svd_pp:
                for item in samples.iterkeys():
                    y[item] += gamma * (s - Lambda * y[item])
        gamma *= slow_rate
        print "step: %s, rmse: %s, mae: %s" % (step + 1, numpy.sqrt(rmse_sum / _tot), mae_sum / _tot)

#生成负样本
def __random_negative_sample(items, ratio=1):
    """
    :param items: 正样本
    :param ratio: 负样本/正样本比例
    :return: 所有样本
    """
    ret = {}
    for item in items.iterkeys():
        ret[item] = 1
    n = 0
    items_len = len(items)
    for _ in xrange(items_len * ratio * 2):
        item = _movie_list[int(numpy.random.random() * _tot)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > items_len * ratio:
            break
    # all_tot = len(_movie_set)
    # while n + items_len < all_tot:
    #     item = _movie_list[int(numpy.random.random() * _tot)]
    #     if item in ret:
    #         continue
    #     ret[item] = 0
    #     n += 1
    #     if n > items_len * ratio:
    #         break
    return ret


# def __random_negative_sample_new(items, ratio=1):
#     """
#     生成负样本
#     :param items: 正样本
#     :param ratio: 负样本/正样本比例
#     :return: 所有样本
#     """
#     ret = {}
#     for item in items.iterkeys():
#         ret[item] = 1
#     import data_structure
#     movie_dic = {}
#     movie_idx = {}
#     for item in _movie_list:
#         movie_dic.setdefault(item, 0)
#         movie_dic[item] += 1
#     tot = 0
#     for item in movie_dic.iterkeys():
#         tot += 1
#         movie_idx[tot] = item
#     bit = data_structure.FenwickTree(tot)
#     for idx, item in movie_idx.iteritems():
#         if item not in ret:
#             bit.add(idx, movie_dic[item])
#     n = 0
#     items_len = len(items)
#     cnt = bit.get(tot)
#     while n <= items_len * ratio and cnt > 0:
#         idx = bit.find_kth(int(numpy.random.random() * cnt))
#         ret[movie_idx[idx]] = 0
#         n += 1
#         bit.add(idx, -(bit.get(idx) - bit.get(idx - 1)))
#         cnt = bit.get(tot)
#     return ret


##########核心方法#########################

#1.评分预测1: 用户u对物品i的预测评分 ---单个预测评分
def __predict(user, item):
    """
    :param user: 用户
    :param item: 物品
    :return: 预测评分值
    """
    rui = 0
    if _bias:
        _bu.setdefault(user, 0)
        _bi.setdefault(item, 0)
        rui += _avr + _bu[user] + _bi[item]
    if _svd:
        _pu.setdefault(user, numpy.zeros((_k, 1)))
        _qi.setdefault(item, numpy.zeros((_k, 1)))
        _z.setdefault(user, numpy.zeros((_k, 1)))
        rui += numpy.sum((_pu[user] + _z[user] if _svd_pp else _pu[user]) * _qi[item])
    # if rui > 5:
    #     rui = 5
    # elif rui < 1:
    #     rui = 1
    return rui


#2.评分预测2：用户u对所有物品的评分预测----基于__predict（）的预测评分汇总
def recommend_explicit(user):
    """
    用户u对物品i的评分预测
    :param user: 用户
    :return: 推荐列表
    """
    rank = {}
    ru = _user_items[user]
    for item in _movie_set:
        if item in ru:
            continue
        rank[item] = __predict(user, item)
    return rank.iteritems()

#3.推荐列表top-n
def recommend_implicit(user, n):
    """
    用户u对物品i评分的可能性预测
    :param user: 用户
    :param n: 为用户推荐n个物品
    :return: 推荐列表
    """
    rank = {}
    ru = _user_items[user]
    for item in _movie_set:
        if item in ru:
            continue
        rank[item] = __predict(user, item)
    return heapq.nlargest(n, rank.iteritems(), key=operator.itemgetter(1))#该方法是返回数组中n个最大值，例如，从中找出最大的4个数



