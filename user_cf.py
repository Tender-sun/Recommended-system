# -*- coding: utf-8 -*-

from __future__ import division

import heapq
import math
import operator


#一、数据预处理
def __pre_treat(train, implicit):
    """
    :param train: 训练集
    :param implicit: 训练集类型
    """
    global _user_items, _item_users, _avr
    _user_items = train
    _item_users = {}
    _avr = {}
    for user, items in _user_items.iteritems():
        for item, rating in items.iteritems():
            _item_users.setdefault(item, {})
            _item_users[item][user] = rating
        if not implicit:
            _avr[user] = sum(items.itervalues()) / len(items)
            
            
#二、相似度计算方法*5
            
#1.余弦相似度(常用)
def user_similarity_cosine(train, iif=False, implicit=False):
    """
    通过余弦相似度计算u和v的兴趣相似度
    :param train: 训练集
    :param iif: 是否惩罚热门物品
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    c = {}
    n = {}
    for users in _item_users.itervalues():
        iif_value = math.log(1 + len(users))
        for u, ru in users.iteritems():
            n.setdefault(u, 0)
            n[u] += ru ** 2
            c.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += ru * rv if not iif else ru * rv / iif_value
    global _w
    _w = {}
    for u, related_users in c.iteritems():
        _w[u] = []
        for v, cuv in related_users.iteritems():
            _w[u].append([v, cuv / math.sqrt(n[u] * n[v])])
        if implicit:
            _w[u].sort(key=operator.itemgetter(1), reverse=True)
            

#2.Jaccard
def user_similarity_jaccard(train, iif=False, implicit=False):
    """
    通过Jaccard相似度计算u和v的兴趣相似度
    :param train: 训练集
    :param iif: 是否惩罚热门物品
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    c = {}
    n = {}
    for users in _item_users.itervalues():
        iif_value = math.log(1 + len(users))
        for u, ru in users.iteritems():
            n.setdefault(u, 0)
            n[u] += ru ** 2
            c.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += ru * rv if not iif else ru * rv / iif_value
    global _w
    _w = {}
    for u, related_users in c.iteritems():
        _w[u] = []
        for v, cuv in related_users.iteritems():
            _w[u].append([v, cuv / (n[u] + n[v] - cuv)])
        if implicit:
            _w[u].sort(key=operator.itemgetter(1), reverse=True)
            

#3.皮尔逊相关系数(常用)
def user_similarity_pearson(train, iif=False, implicit=False):
    """
    通过皮尔逊相关系数计算u和v的兴趣相似度
    :param train: 训练集
    :param iif: 是否惩罚热门物品
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    avr_x = {}
    avr_y = {}
    tot = {}
    for users in _item_users.itervalues():
        for u, ru in users.iteritems():
            avr_x.setdefault(u, {})
            avr_y.setdefault(u, {})
            tot.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                avr_x[u].setdefault(v, 0)
                avr_x[u][v] += ru
                avr_y[u].setdefault(v, 0)
                avr_y[u][v] += rv
                tot[u].setdefault(v, 0)
                tot[u][v] += 1
    for u, related_users in tot.iteritems():
        for v, cnt in related_users.iteritems():
            avr_x[u][v] /= cnt
            avr_y[u][v] /= cnt
    c = {}
    x = {}
    y = {}
    for users in _item_users.itervalues():
        iif_value = math.log(1 + len(users))
        for u, ru in users.iteritems():
            c.setdefault(u, {})
            x.setdefault(u, {})
            y.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += (ru - avr_x[u][v]) * (rv - avr_y[u][v]) if not iif else (ru - avr_x[u][v]) * (
                    rv - avr_y[u][v]) / iif_value
                x[u].setdefault(v, 0)
                x[u][v] += (ru - avr_x[u][v]) ** 2
                y[u].setdefault(v, 0)
                y[u][v] += (rv - avr_y[u][v]) ** 2
    global _w
    _w = {}
    for u, related_users in c.iteritems():
        _w[u] = []
        for v, cuv in related_users.iteritems():
            _w[u].append([v, cuv / math.sqrt(x[u][v] * y[u][v]) if x[u][v] * y[u][v] else 0])
        if implicit:
            _w[u].sort(key=operator.itemgetter(1), reverse=True)
            

#4.基于调整的余弦相似度
def user_similarity_adjusted_cosine(train, iif=False, implicit=False):
    """
    通过余弦相似度计算u和v的兴趣相似度
    :param train: 训练集
    :param iif: 是否惩罚热门物品
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    c = {}
    n = {}
    for users in _item_users.itervalues():
        iif_value = math.log(1 + len(users))
        for u, ru in users.iteritems():
            n.setdefault(u, 0)
            n[u] += (ru - _avr[u]) ** 2#主要调整：ru - _avr[u] //   rv - _avr[v] (_avr源于pre_treat)
            c.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += (ru - _avr[u]) * (rv - _avr[v]) if not iif else (ru - _avr[u]) * (rv - _avr[v]) / iif_value
    global _w
    _w = {}
    for u, related_users in c.iteritems():
        _w[u] = []
        for v, cuv in related_users.iteritems():
            _w[u].append([v, cuv / math.sqrt(n[u] * n[v]) if n[u] * n[v] else 0])
        if implicit:
            _w[u].sort(key=operator.itemgetter(1), reverse=True)
            

#5.对数似然比
def user_similarity_log_likelihood(train, implicit=True):
    """
    通过对数似然比计算u和v的兴趣相似度
    :param train: 训练集
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    c = {}
    n = {}
    for users in _item_users.itervalues():
        for u, ru in users.iteritems():
            n.setdefault(u, 0)
            n[u] += ru ** 2
            c.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += ru * rv
    global _w
    _w = {}
    item_cnt = len(_item_users)
    for u, related_users in c.iteritems():
        _w[u] = []
        for v, cuv in related_users.iteritems():
            _w[u].append([v, __calc_log_likelihood(cuv, n[u] - cuv, n[v] - cuv, item_cnt - n[u] - n[v] + cuv)])
        if implicit:
            _w[u].sort(key=operator.itemgetter(1), reverse=True)
            


def __calc_log_likelihood(num_both, num_x, num_y, num_none):
    """
    :param num_both: x和y共同偏好的数量
    :param num_x: x单独偏好的数量
    :param num_y: y单独偏好的数量
    :param num_none: x和y都不偏好的数量
    :return: 对数似然比
    """
    p1 = num_both / (num_both + num_x)
    p2 = num_y / (num_y + num_none)
    p = (num_both + num_y) / (num_both + num_x + num_y + num_none)
    r1 = 0
    r2 = 0
    if 0 < p <= 1:
        r1 += num_both * math.log(p) + num_y * math.log(p)
    if 0 <= p < 1:
        r1 += num_x * math.log(1 - p) + num_none * math.log(1 - p)
    if 0 < p1 <= 1:
        r2 += num_both * math.log(p1)
    if 0 <= p1 < 1:
        r2 += num_x * math.log(1 - p1)
    if 0 < p2 <= 1:
        r2 += num_y * math.log(p2)
    if 0 <= p2 < 1:
        r2 += num_none * math.log(1 - p2)
    return 2 * (r2 - r1)



#&&&三、核心方法：进行推荐&&&

def recommend_explicit(user):#用户可能评多少分
    """
    用户u对物品i的评分预测
    :param user: 用户
    :return: 推荐列表
    """
    rank = {}
    w_sum = {}
    ru = _user_items[user]
    for v, wuv in _w[user]:
        for i, rvi in _user_items[v].iteritems():
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += wuv * (rvi - _avr[v])
            w_sum.setdefault(i, 0)
            w_sum[i] += abs(wuv)
    for item in rank.iterkeys():
        if w_sum[item]:
            rank[item] /= w_sum[item]
        rank[item] += _avr[user]
    return rank.iteritems()


def recommend_implicit(user, n, k):#用户会不会评分
    """
    用户u对物品i评分的可能性预测
    :param user: 用户
    :param n: 为用户推荐n个物品
    :param k: 取和用户u兴趣最接近的k个用户
    :return: 推荐列表
    """
    cnt = {}
    rank = {}
    ru = _user_items[user]
    for v, wuv in _w[user]:
        for i, rvi in _user_items[v].iteritems():
            if i in ru:
                continue
            cnt.setdefault(i, 0)
            if cnt[i] == k:
                continue
            cnt[i] += 1
            rank.setdefault(i, 0)
            rank[i] += wuv * rvi
    return heapq.nlargest(n, rank.iteritems(), key=operator.itemgetter(1))
