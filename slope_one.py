# -*- coding: utf-8 -*-

from __future__ import division


#1.计算物品i和j的差值
def item_deviation(train, implicit=False):
    """
    :param train: 训练集
    :param implicit: 训练集类型
    """
    global _freq, _user_items
    _user_items = train
    _freq = {}
    deviation = {}
    for items in _user_items.itervalues():
        for i, ri in items.iteritems():
            _freq.setdefault(i, {})
            deviation.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                deviation[i].setdefault(j, 0)
                deviation[i][j] += ri - rj
                _freq[i].setdefault(j, 0)
                _freq[i][j] += 1
    global _w
    _w = {}
    for i, related_items in deviation.iteritems():
        _w[i] = {}
        for j, dij in related_items.iteritems():
            _w[i][j] = dij / _freq[i][j]


#2.用户u对物品i的评分预测
def recommend_explicit(user):
    """
    :param user: 用户
    :return: 推荐列表
    """
    rank = {}
    freq_sum = {}
    ru = _user_items[user]
    for j, ruj in ru.iteritems():
        for i, wji in _w[j].iteritems():
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += (ruj - wji) * _freq[j][i]  # wij == -wji
            freq_sum.setdefault(i, 0)
            freq_sum[i] += _freq[j][i]  # _freq[i][j] == _freq[j][i]，但后者对cache更友好
    for item in rank.iterkeys():
        if freq_sum[item]:
            rank[item] /= freq_sum[item]
    return rank.iteritems()




