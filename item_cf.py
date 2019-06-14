# -*- coding: utf-8 -*-

from __future__ import division

import heapq
import math
import operator

#数据预处理
def __pre_treat(train, implicit):
    """
    :param train: 训练集
    :param implicit: 训练集类型
    """
    global _user_items, _item_users, _avr
    _user_items = train
    _item_users = {}
    for user, items in _user_items.iteritems():
        for item, rating in items.iteritems():
            _item_users.setdefault(item, {})
            _item_users[item][user] = rating
    if not implicit:
        _avr = {}
        for item, users in _item_users.iteritems():
            _avr[item] = sum(users.itervalues()) / len(users)


#相似度计算方法
            
#1.余弦相似度(常用)
def item_similarity_cosine(train, norm=False, iuf=False, implicit=False):
    """
    通过余弦相似度计算物品i和j的相似度
    :param train: 训练集
    :param norm: 是否归一化
    :param iuf: 是否惩罚热门物品
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    c = {}
    n = {}
    for items in _user_items.itervalues():
        iuf_value = math.log(1 + len(items))
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += ri ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += ri * rj if not iuf else ri * rj / iuf_value
    global _w
    _w = {}
    for i, related_items in c.iteritems():
        _w[i] = []
        for j, cij in related_items.iteritems():
            _w[i].append([j, cij / math.sqrt(n[i] * n[j])])
        if norm:
            norm_value = max(abs(item[1]) for item in _w[i])
            if norm_value:
                for item in _w[i]:
                    item[1] /= norm_value
        if implicit:
            _w[i].sort(key=operator.itemgetter(1), reverse=True)

#2.Jaccard
def item_similarity_jaccard(train, norm=False, iuf=False, implicit=False):
    """
    通过Jaccard相似度计算物品i和j的相似度
    :param train: 训练集
    :param norm: 是否归一化
    :param iuf: 是否惩罚热门物品
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    c = {}
    n = {}
    for items in _user_items.itervalues():
        iuf_value = math.log(1 + len(items))
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += ri ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += ri * rj if not iuf else ri * rj / iuf_value
    global _w
    _w = {}
    for i, related_items in c.iteritems():
        _w[i] = []
        for j, cij in related_items.iteritems():
            _w[i].append([j, cij / (n[i] + n[j] - cij)])
        if norm:
            norm_value = max(abs(item[1]) for item in _w[i])
            if norm_value:
                for item in _w[i]:
                    item[1] /= norm_value
        if implicit:
            _w[i].sort(key=operator.itemgetter(1), reverse=True)

#3.皮尔逊相关系数(常用)---代码量较USERCF多一点
def item_similarity_pearson(train, norm=False, iuf=False, implicit=False):
    """
    通过皮尔逊相关系数计算物品i和j的相似度
    :param train: 训练集
    :param norm: 是否归一化
    :param iuf: 是否惩罚热门物品
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    avr_x = {}
    avr_y = {}
    tot = {}
    for items in _user_items.itervalues():
        for i, ri in items.iteritems():
            avr_x.setdefault(i, {})
            avr_y.setdefault(i, {})
            tot.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                avr_x[i].setdefault(j, 0)
                avr_x[i][j] += ri
                avr_y[i].setdefault(j, 0)
                avr_y[i][j] += rj
                tot[i].setdefault(j, 0)
                tot[i][j] += 1
    for i, related_items in tot.iteritems():
        for j, cnt in related_items.iteritems():
            avr_x[i][j] /= cnt
            avr_y[i][j] /= cnt
    c = {}
    x = {}
    y = {}
    for items in _user_items.itervalues():
        iuf_value = math.log(1 + len(items))
        for i, ri in items.iteritems():
            c.setdefault(i, {})
            x.setdefault(i, {})
            y.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += (ri - avr_x[i][j]) * (rj - avr_y[i][j]) if not iuf else (ri - avr_x[i][j]) * (
                    rj - avr_y[i][j]) / iuf_value
                x[i].setdefault(j, 0)
                x[i][j] += (ri - avr_x[i][j]) ** 2
                y[i].setdefault(j, 0)
                y[i][j] += (rj - avr_y[i][j]) ** 2
    global _w
    _w = {}
    for i, related_items in c.iteritems():
        _w[i] = []
        for j, cij in related_items.iteritems():
            _w[i].append([j, cij / math.sqrt(x[i][j] * y[i][j]) if x[i][j] * y[i][j] else 0])
        if norm:
            norm_value = max(abs(item[1]) for item in _w[i])
            if norm_value:
                for item in _w[i]:
                    item[1] /= norm_value
        if implicit:
            _w[i].sort(key=operator.itemgetter(1), reverse=True)

#4.基于调整的余弦相似度
def item_similarity_adjusted_cosine(train, norm=False, iuf=False, implicit=False):
    """
    通过余弦相似度计算物品i和j的相似度
    :param train: 训练集
    :param norm: 是否归一化
    :param iuf: 是否惩罚热门物品
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    c = {}
    n = {}
    for user, items in _user_items.iteritems():
        iuf_value = math.log(1 + len(items))
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += (ri - _avr[i]) ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)#主要调整：ri - _avr[i]  rj - _avr[j]
                c[i][j] += (ri - _avr[i]) * (rj - _avr[j]) if not iuf else (ri - _avr[i]) * (rj - _avr[j]) / iuf_value
    global _w
    _w = {}
    for i, related_items in c.iteritems():
        _w[i] = []
        for j, cij in related_items.iteritems():
            _w[i].append([j, cij / math.sqrt(n[i] * n[j]) if n[i] * n[j] else 0])
        if norm:
            norm_value = max(abs(item[1]) for item in _w[i])
            if norm_value:
                for item in _w[i]:
                    item[1] /= norm_value
        if implicit:
            _w[i].sort(key=operator.itemgetter(1), reverse=True)

#5.对数似然比
def item_similarity_log_likelihood(train, norm=False, implicit=True):
    """
    通过对数似然比计算物品i和j的相似度
    :param train: 训练集
    :param norm: 是否归一化
    :param implicit: 训练集类型
    """
    __pre_treat(train, implicit)
    c = {}
    n = {}
    for items in _user_items.itervalues():
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += ri ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += ri * rj
    global _w
    _w = {}
    user_len = len(train)
    for i, related_items in c.iteritems():
        _w[i] = []
        for j, cij in related_items.iteritems():
            _w[i].append([j, __calc_log_likelihood(cij, n[i] - cij, n[j] - cij, user_len - n[i] - n[j] + cij)])
        if norm:
            norm_value = max(abs(item[1]) for item in _w[i])
            if norm_value:
                for item in _w[i]:
                    item[1] /= norm_value
        _w[i].sort(key=operator.itemgetter(1), reverse=True)


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


#################  &&&核心方法&&&

def recommend_explicit(user):#用户可能评多少分
    """
    用户u对物品i的评分预测
    :param user: 用户
    :return: 推荐列表
    """
    rank = {}
    w_sum = {}
    ru = _user_items[user]
    for j, ruj in ru.iteritems():
        for i, wji in _w[j]:
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += wji * (ruj - _avr[j])
            w_sum.setdefault(i, 0)
            w_sum[i] += abs(wji)
    for item in rank.iterkeys():
        if w_sum[item]:
            rank[item] /= w_sum[item]
        rank[item] += _avr[item]
    return rank.iteritems()


def recommend_implicit(user, n, k):#用户会不会评分
    """
    用户u对物品i评分的可能性预测
    :param user: 用户
    :param n: 为用户推荐n个物品
    :param k: 取和物品j最相似的k个物品
    :return: 推荐列表
    """
    rank = {}
    ru = _user_items[user]
    for j, ruj in ru.iteritems():
        for i, wji in _w[j][:k]:
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += wji * ruj
    return heapq.nlargest(n, rank.iteritems(), key=operator.itemgetter(1))





