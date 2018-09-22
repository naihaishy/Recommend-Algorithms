# -*- coding:utf-8 -*-
# @Time : 2018/9/21 11:27
# @Author : naihai

import utils

"""
基于用户的协同过滤算法
最简单的UserCF
"""

if __name__ == '__main__':
    train, test = utils.split_data(utils.load_data("./data/ratings.dat"), 8, 1)
