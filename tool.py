# -*- coding: utf-8 -*-

#工具类，不常用
class FenwickTree(object):
    def __init__(self, length):
        self.n = length
        self.bit = [0] * (self.n + 1)

    @classmethod
    def lowbit(cls, num):
        return num & -num

    def add(self, index, value):
        while index <= self.n:
            self.bit[index] += value
            index += self.lowbit(index)

    def get(self, index):
        ret = 0
        while index > 0:
            ret += self.bit[index]
            index -= self.lowbit(index)
        return ret

    def find_kth(self, k):
        ret = 0
        cnt = 0
        index = 1
        while index <= self.n:
            index <<= 1
        while index > 0:
            ret += index
            if ret >= self.n or cnt + self.bit[ret] >= k:
                ret -= index
            else:
                cnt += self.bit[ret]
            index >>= 1
        return ret + 1
