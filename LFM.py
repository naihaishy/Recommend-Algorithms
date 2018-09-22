# -*- coding:utf-8 -*-
# @Time : 2018/9/22 17:55
# @Author : naihai

import utils
"""
隐语义模型 LFM
"""

if __name__ == '__main__':
    train, test = utils.split_data(utils.load_data("./data/ratings.dat"), 8, 1)