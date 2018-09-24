# -*- coding:utf-8 -*-
# @Time : 2018/9/21 9:13
# @Author : naihai

import math
import time
import operator
import utils

"""
基于用户的协同过滤算法
最简单的UserCF
"""


def user_similarity_single_thread(_train, iif=False):
    """
    常规计算用户相似度
    :param _train:
    :param iif: bool 是否考虑物品流行度
    :return:
    """
    # 统计矩阵
    user_sims = dict()
    co_users = dict()  # 表示两个用户共同交互过多少物品
    item_users = dict()  # 每个物品所属的用户集合
    user_items = dict()  # 每个用户交互过的物品数目

    for user, items in _train.items():
        user_items[user] = len(items)
        for item in items:
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(user)  # 拥有该物品的用户集合

    for item, users in item_users.items():
        if iif:
            iif_value = 1 / math.log(1 + len(users))
        else:
            iif_value = 1
        # users 拥有该物品的用户集合
        for user_u in users:
            if user_u not in co_users:
                co_users[user_u] = dict()
            for user_v in users:
                if user_v not in co_users[user_u]:
                    co_users[user_u][user_v] = 0
                co_users[user_u][user_v] += iif_value

    # 计算用户之间的相似度 只有拥有共同物品的用户才有相似度 这降低了计算量
    for user_u, related_users in co_users.items():
        if user_u not in user_sims:
            user_sims[user_u] = dict()
        # u用户与其他用户
        for user_v, cuv in related_users.items():
            # cuv表示u v用户共有多少物品
            user_sims[user_u][user_v] = cuv / math.sqrt(user_items[user_u] * user_items[user_v])
    return user_sims


def user_similarity_iif(_train):
    """
    User-IIF 相似度计算的改进
    Inverse User Frequency 物品流行度对数的倒数的参数
    热门的物品对用户相似度的贡献应当降低
    :param _train: dict 2d
    :return:
    """
    user_sims = user_similarity_single_thread(_train, True)
    return user_sims


def user_similarity_normal(_user_sims):
    """
    对用户相似度矩阵最大值归一化
    :param _user_sims:
    :return:
    """
    for user_u, related_users in _user_sims.items():
        # u用户与其他用户的相似度
        sims = [sim_uv for (_, sim_uv) in related_users.items()]
        max_sim = max(sims)

        for user_v, sim_uv in related_users.items():
            _user_sims[user_u][user_v] /= max_sim


def user_similarity(train_data):
    """
    计算用户之间的相似度
    :param train_data:
    :return:
    """
    start = time.time()
    user_sims = user_similarity_single_thread(train_data)
    # user_sims = user_similarity_iif(train_data)

    # compare_user_similarity(user_sims, user_sims1)

    # 归一化
    # user_similarity_normal(user_sims)
    print("user similarity done, cost " + str(time.time() - start) + " s")
    return user_sims


def recommend_single_thread(_train, _user_sims, _nearest_k, _top_n):
    """
    单线程推荐
    :param _train: dict{list} 训练集
    :param _user_sims: dict 2d 用户相似度矩阵
    :param _nearest_k: int 使用最近的k个用户计算评分
    :param _top_n: int 返回评分最高的n个用户
    :return recommend_lists: dict{list} 每个用户的top n推荐列表
    """
    recommend_lists = dict()

    for user_u, interacted_items in _train.items():
        rank = dict()
        # 根据相似度矩阵, 找到距离用户u最近的K个用户
        nearest_users = sorted(_user_sims[user_u].items(), key=operator.itemgetter(1), reverse=True)[0:_nearest_k]
        for user_v, wuv in nearest_users:
            # 用户v交互过的物品
            user_v_interacted_items = _train.get(user_v)
            for item, rating in user_v_interacted_items.items():
                if item in interacted_items:
                    continue  # 该物品已经被用户u交互过了
                if item not in rank:
                    rank[item] = 0
                rank[item] += rating * wuv  # # rating是用户对该物品的评分 wuv是用户uv的相似度

        # 选择Top N 作为该用户的推荐
        top_n_items = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:_top_n]
        # 转成list
        recommend_lists[user_u] = [user for (user, _) in top_n_items]
    return recommend_lists


def recommend(train_data, user_sims, nearest_k=5, top_n=10):
    """
    执行推荐 为每个用户都推荐用户
    :param train_data: dict{list} 训练集
    :param user_sims: dict 2d 用户相似度矩阵
    :param nearest_k: int 使用最近的k个用户计算评分
    :param top_n: int 返回评分最高的n个用户
    :return recommend_lists: dict{list} 每个用户的top n推荐列表
    """
    start = time.time()
    recommend_list = recommend_single_thread(train_data, user_sims, nearest_k, top_n)
    print("recommend done, cost " + str(time.time() - start) + " s")
    return recommend_list


if __name__ == '__main__':

    start_time = time.time()

    train, test = utils.split_data(utils.load_data("./data/ratings.dat"), 8, 1)

    W = user_similarity(train)

    recommends = recommend(train, W, nearest_k=80, top_n=10)

    p = utils.precision(train, test, recommends)

    r = utils.recall(train, test, recommends)

    c = utils.coverage(train, recommends)

    po = utils.popularity(train, recommends)

    cost_time = time.time() - start_time

    print(p, r, c, po)
    print("cost time " + str(cost_time) + " s")
