# -*- coding:utf-8 -*-
# @Time : 2018/9/22 17:55
# @Author : naihai
import random
import math
import time
import operator

import utils

"""
隐语义模型 LFM
"""


def random_generate_samples(interacted_items, items_pool):
    """
    对该用户生成样本集
    对于每个用户 正负样本数目相同
    正样本 用户交互过的即为正样本
    负样本 选择那些热门 但是用户没有交互的物品
    :param interacted_items: 该用户交互过的项目即为正样本
    :param items_pool: 所有的物品集合 允许重复
    :return:
    """
    sample_nums = len(interacted_items)  # 正负样本数目
    negative_samples = dict()

    positive_samples = {key: 1 for key in interacted_items}  # 正样本
    # 均匀采样负样本
    n = 0
    while True:
        item = items_pool[random.randint(0, len(items_pool) - 1)]  # 随机采样
        if item in negative_samples or item in positive_samples:
            continue
        negative_samples[item] = 0
        n += 1
        if n == sample_nums:
            break
    samples = {**positive_samples, **negative_samples}  # 所有样本集合
    return samples


def init_model(train_data, factor_nums):
    """
    随机初始化 P Q 矩阵 每个值为0-1之间
    :param train_data:
    :param factor_nums:
    :return:
    """
    start = time.time()
    P = dict()
    Q = dict()
    all_users = [user for user in train_data]
    all_items = list(set([item for (_, items) in train_data.items() for item in items]))

    for user in all_users:
        if user not in P:
            P[user] = dict()
        for f in range(factor_nums):
            P[user][f] = random.random()
    for item in all_items:
        if item not in Q:
            Q[item] = dict()
        for f in range(factor_nums):
            Q[item][f] = random.random()
    print("init model done, cost " + str(time.time() - start) + " s")
    return [P, Q]


def prediction(pu_dict, qi_dict):
    """
    预测用户u对物品i的评分
    :param pu_dict: 用户u的隐向量
    :param qi_dict: 物品i的隐向量
    :return:
    """
    rank = 0
    for f, puf in pu_dict.items():
        rank += puf * qi_dict[f]
    rank = 1.0 / (1 + math.exp(-rank))  # sigmoid 0-1
    return rank


def learning(train_data, F, N, alpha, flambda):
    """
    优化过程 SGD
    :return:
    """
    start = time.time()
    # 初始化模型参数
    [P, Q] = init_model(train_data, F)

    # 根据流向度构建待选样本集合 所有的物品集合
    items_pool = [item for (_, items) in train_data.items() for item in items]

    # 迭代过程
    for i in range(N):
        for user, items in train_data.items():
            samples = random_generate_samples(items, items_pool)
            for item, rui in samples.items():
                eui = rui - prediction(P[user], Q[item])  # 用户user对item的评分预测与真实值之间的差值
                # F个因子factor
                for f in range(F):
                    # 更新参数
                    P[user][f] += alpha * (Q[item][f] * eui - flambda * P[user][f])
                    Q[item][f] += alpha * (P[user][f] * eui - flambda * Q[item][f])
        # 更新学习率
        alpha *= 0.9
        print("iter", i, "done")

    print("learning done, cost " + str(time.time() - start) + " s")
    return [P, Q]


def recommend(train_data, p_latents, q_latents, top_n):
    """
    执行推荐 为每个用户都推荐物品
    :return:
    """
    start = time.time()

    recommend_lists = dict()

    for user, interacted_items in train_data.items():
        rank = dict()  # 用户u对所有物品的评分矩阵
        for f, puf in p_latents[user].items():
            for item in q_latents:
                if item not in rank:
                    rank[item] = 0
                rank[item] += puf * q_latents[item][f]

        # 选择Top N 作为该用户的推荐
        top_n_items = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:top_n]
        # 转成list
        recommend_lists[user] = [item for (item, _) in top_n_items]
    print("recommend done, cost " + str(time.time() - start) + " s")
    return recommend_lists


if __name__ == '__main__':

    start_time = time.time()

    train, test = utils.split_data(utils.load_data("./data/ratings.dat"), 10, 2)

    [P, Q] = learning(train, 100, 20, 0.02, 0.01)

    recommends = recommend(train, P, Q, top_n=10)

    p = utils.precision(train, test, recommends)

    r = utils.recall(train, test, recommends)

    c = utils.coverage(train, recommends)

    po = utils.popularity(train, recommends)

    cost_time = time.time() - start_time

    print(p, r, c, po)
    print("cost time " + str(cost_time) + " s")
