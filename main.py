# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import method

'''
def test100k_explicit():
    test_count = 5#测试集：
    evaluation_base = 2#训练集：
    ans = [0] * evaluation_base
    #print ans
    for k in xrange(1, test_count + 1):#xrange(1,6)循环1到5，处理较大的数字序列（对比range）
        #print k
        method.generate_data_100k_explicit(k)#1.通过for选择u1-u5数据文件
        #print 'a'
        #print a
        method.generate_matrix(implicit=False)#2.回调生成评分矩阵的函数
        #print 'x'
        #print x 
        b = method.evaluate_explicit()#3.评估算法之RMSE和MAE
        print k
        print b
        #print 'b'
        #print b
        for x in xrange(evaluation_base):#4.在不同K情况下，将RMSE和MAE存到ans中
            #print 'x'
            #print x
            ans[x] += b[x]
            #print 'ans'
            #print ans
            
    for x in xrange(evaluation_base):#因为K变化，每次的RMSE和MAE不同，一共5组值；通过循环求平均值
        #print "x"
        #print x
        ans[x] /= test_count
        #print 'ans'
        #print ans
    print ans


def test1m_explicit():
    test_count = 8
    evaluation_base = 2
    ans = [0] * evaluation_base
    for k in xrange(test_count):
        method.generate_data_1m_explicit(test_count, k)
        method.generate_matrix(implicit=False)
        b = method.evaluate_explicit()
        for x in xrange(evaluation_base):
            ans[x] += b[x]
    for x in xrange(evaluation_base):
        ans[x] /= test_count
    print ans


def test_latest_small_explicit():
    test_count = 8
    evaluation_base = 2
    ans = [0] * evaluation_base
    for k in xrange(test_count):
        method.generate_data_latest_small_explicit(test_count, k)
        method.generate_matrix(implicit=False)
        b = method.evaluate_explicit()
        for x in xrange(evaluation_base):
            ans[x] += b[x]
    for x in xrange(evaluation_base):
        ans[x] /= test_count
    print ans

    

'''
# *_implicit: 用户u对物品i评分的可能性预测---是否评

def test100k_implicit():
    test_count = 5
    evaluation_base = 4#&&&不同点1
    ans = [0] * evaluation_base
    for k in xrange(1, test_count + 1):
        method.generate_data_100k_implicit(k)#&&&不同点2
        method.generate_matrix(implicit=True)#&&&不同点3e
        b = method.evaluate_implicit()#&&&不同点4
        for x in xrange(evaluation_base):
            ans[x] += b[x]
    for x in xrange(evaluation_base):
        ans[x] /= test_count
    print ans


def test1m_implicit():
    test_count = 8
    evaluation_base = 4
    ans = [0] * evaluation_base
    for k in xrange(test_count):
        method.generate_data_1m_implicit(test_count, k)
        method.generate_matrix(implicit=True)
        b = method.evaluate_implicit()
        for x in xrange(evaluation_base):
            ans[x] += b[x]
    for x in xrange(evaluation_base):
        ans[x] /= test_count
    print ans


def test_latest_small_implicit():
    test_count = 8
    evaluation_base = 4
    ans = [0] * evaluation_base
    for k in xrange(test_count):
        method.generate_data_latest_small_implicit(test_count, k)
        method.generate_matrix(implicit=True)
        b = method.evaluate_implicit()
        for x in xrange(evaluation_base):
            ans[x] += b[x]
    for x in xrange(evaluation_base):
        ans[x] /= test_count
    print ans

'''



if __name__ == '__main__':

   
    #注意:修改method中对应算法，否则始终测试默认的USERCF！！！
    
    #print '100K---'
    test100k_explicit() 
    #
    
    #print '1m---'
    #test1m_explicit() #
    #test1m_implicit() #

    #print 'latest_small---'
    #test_latest_small_explicit() #
    #test_latest_small_implicit() #

    
    

