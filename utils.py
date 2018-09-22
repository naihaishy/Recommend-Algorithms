# -*- coding:utf-8 -*-
# @Time : 2018/9/22 18:07
# @Author : naihai

import time
import random

"""
常用工具函数
"""


def split_data(all_data, m, k):
    """
    切分数据集 切为M分 k分作为测试集 剩下的作为训练集
    :param all_data: dict 2d
    :param m:
    :param k:
    :return:
    """
    start = time.time()

    train_data = dict()
    test_data = dict()

    random.seed(time.time())

    for user, items in all_data.items():
        # 初始化dict
        if user not in train_data:
            train_data[user] = dict()
        if user not in test_data:
            test_data[user] = dict()

        # 下面random sample方式效率明显高于 随机采样 if random.int(0,m) == k
        test_data_keys = random.sample(list(items), int(len(items) / m * k))
        train_data_keys = set(items.keys()) - set(test_data_keys)

        for key in test_data_keys:
            test_data[user][key] = items[key]

        for key in train_data_keys:
            train_data[user][key] = items[key]

    print("split data done, cost " + str((time.time() - start) * 1000) + " ms")
    return train_data, test_data


def load_data(file):
    """
    加载数据
    :param file:
    :return: all_data dict 2d
    """
    start = time.time()
    all_data = dict()
    with open(file, 'r') as f:
        line = f.readline()
        while line is not None and line != '':
            arr = line.split("::")
            user = int(arr[0])
            item = int(arr[1])
            rating = int(arr[2])
            if user not in all_data:
                all_data[user] = dict()
            all_data[user][item] = rating
            line = f.readline()
    print("load data done, cost " + str((time.time() - start) * 1000) + " ms")
    return all_data
    return_data = dict()
    for key in all_data.keys():
        if key <= 10000:
            return_data[key] = all_data.get(key)
    return return_data


def save_result(_cost_time, _nearest_k, _top_n, _p, _r, _c, _po):
    """
    保存结果到文件中
    :param _cost_time:
    :param _nearest_k:
    :param _top_n:
    :param _p:
    :param _r:
    :param _c:
    :param _po:
    :return:
    """
    with open('result.txt', 'a') as f:
        f.write(str(_nearest_k) + ", " + str(_top_n) + ", " + str(_p) + ", " + str(_r)
                + ", " + str(_c) + ", " + str(_po) + "\n")


def plot_result():
    pass


def precision(train_data, test_data, recommend_list):
    """
    计算precision准确度
    :param train_data: dict{list} 训练集
    :param test_data: dict{list} 测试集
    :param recommend_list: dict{list} 每个用户的top n推荐列表
    :return:
    """
    start = time.time()
    hit_num = 0
    all_num = 0

    for user in train_data.keys():
        tu = test_data[user]  # 测试集中该用户交互的物品
        rank = recommend_list.get(user)  # 给该用户推荐的物品
        for item in rank:
            if item in tu:
                hit_num += 1
        all_num += len(rank)
    print("calculate precision done, cost " + str(time.time() - start) + " s")
    return hit_num / (all_num * 1.0)


def recall(train_data, test_data, recommend_list):
    """
    计算recall召回率
    :param train_data:
    :param test_data:
    :param recommend_list:
    :return:
    """
    start = time.time()
    hit_num = 0
    all_num = 0
    for user in train_data.keys():
        tu = test_data[user]
        rank = recommend_list.get(user)  # 给该用户推荐的物品
        for item in rank:
            if item in tu:
                hit_num += 1
        all_num += len(tu)
    print("calculate recall done, cost " + str(time.time() - start) + " s")
    return hit_num / (all_num * 1.0)


def coverage(train_data, recommend_list):
    """
    计算coverage覆盖率
    :param train_data:
    :param recommend_list:
    :return:
    """
    start = time.time()
    # 推荐的所有物品集合
    recommend_items = set()
    all_items = set()
    for user, items in train_data.items():
        for item in items:
            all_items.add(item)
        rank = recommend_list.get(user)  # 给该用户推荐的物品
        for item in rank:
            recommend_items.add(item)
    print("calculate coverage done, cost " + str(time.time() - start) + " s")
    return len(recommend_items) / (len(all_items) * 1.0)


def popularity(train_data, recommend_list):
    """
    计算popularity流行度
    :param train_data:
    :param recommend_list:
    :return:
    """
    start = time.time()
    item_popularity = dict()
    # 统计流行度
    for user, items in train_data.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1

    # 计算推荐项目的平均流行度
    ret = 0.0
    n = 0
    for user in train_data.keys():
        rank = recommend_list.get(user)  # 给该用户推荐的物品
        for item in rank:
            ret += math.log(1 + item_popularity[item])
        n += len(rank)
    ret /= n * 1.0
    print("calculate popularity done, cost " + str(time.time() - start) + " s")
    return ret
