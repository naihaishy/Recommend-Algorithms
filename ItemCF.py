# -*- coding:utf-8 -*-
# @Time : 2018/9/19 9:13
# @Author : naihai

import math
import time
import operator
import threading
import multiprocessing
from collections import Counter

import utils

"""
基于物品的协同过滤算法
最简单的ItemCF
"""


def item_similarity_multi_process_func1(_train, _sub_users, _result_mq, _iuf=False):
    """
    统计共现矩阵以及每个物品被多少用户交互过
    :param _train:
    :param _sub_users:
    :param _result_mq:
    :param _iuf:
    :return:
    """
    num_items = dict()
    co_items = dict()
    for user in _sub_users:
        items = _train.get(user)  # 该用户交互过的物品
        if _iuf:
            iuf_value = 1 / math.log(1 + len(items) * 1.0)  # IUF 该用户对物品相似度的贡献
        else:
            iuf_value = 1
        for item_i in items:
            if item_i not in co_items:
                co_items[item_i] = dict()
            if item_i not in num_items:
                num_items[item_i] = 0
            num_items[item_i] += 1

            for item_j in items:
                if item_i == item_j:
                    continue
                if item_j not in co_items[item_i]:
                    co_items[item_i][item_j] = 0
                co_items[item_i][item_j] += iuf_value
    _result_mq.put((num_items, co_items))


def item_similarity_multi_process_func2(_co_items, _num_items, _sub_items, _result_mq):
    """
    计算相似度
    :param _sub_items:
    :param _co_items:
    :param _num_items:
    :param _result_mq:
    :return:
    """
    _item_sims = dict()
    for item_i in _sub_items:
        related_items = _co_items.get(item_i)  # 与该物品共同被其他用户交互的所有物品
        if item_i not in _item_sims:
            _item_sims[item_i] = dict()
        # i物品与其他物品
        for item_j, cij in related_items.items():
            # j物品 cij表示物品i和物品j被多少用户共有
            _item_sims[item_i][item_j] = cij / math.sqrt(_num_items[item_i] * _num_items[item_j])
    _result_mq.put(_item_sims)


def item_similarity_multi_process(_train, iuf=False):
    """
    多进程计算物品之间的相似度
    :param _train: dict{list} 训练集
    :param iuf: bool 是否考虑用户活跃度
    :return:
    """

    item_sims = dict()  # 物品相似度矩阵
    co_items = dict()  # 表示两个物品被多少用户共同交互过
    num_items = dict()  # 表示每个用品被多少用户交互过

    cpu_nums = multiprocessing.cpu_count()

    # 统计矩阵 ---------
    users = [user for user in _train]  # 所有用户集合
    user_counts = len(users)

    result_mq_list = []
    process_list = []
    for i in range(cpu_nums):
        # 每个子进程处理一部分数据
        step = int(user_counts / cpu_nums) + 1
        start_idx = step * i
        end_idx = step * (i + 1)
        if end_idx >= user_counts:
            end_idx = user_counts
        sub_users = users[start_idx:end_idx]

        # 进程通信 Queue
        result_mq = multiprocessing.Queue()
        process = multiprocessing.Process(target=item_similarity_multi_process_func1,
                                          args=(_train, sub_users, result_mq, iuf))
        process.start()
        process_list.append(process)
        result_mq_list.append(result_mq)

    # 等待子进程结束

    # 所有进程均执行完毕

    # 所有进程均执行完毕

    # 获取数据
    for queue in result_mq_list:
        (sub_num_items, sub_co_items) = queue.get()
        # 合并dict 相同key的value叠加
        num_items = dict(Counter(num_items) + Counter(sub_num_items))

        # 合并dict 2d 相同key的value叠加
        if co_items is None:
            co_items = sub_co_items
        else:
            for item_i, sub_related_items in sub_co_items.items():
                if item_i in item_sims:
                    # 已经存在该物品为中心的共现dict 合并
                    co_items[item_i] = dict(Counter(co_items[item_i]) + Counter(sub_related_items))
                else:
                    # 尚不存在该物品 赋值
                    co_items[item_i] = sub_related_items

    # 计算物品之间的相似度 ------
    all_items = [item for item in co_items]  # 所有物品集合
    item_counts = len(all_items)  # 物品总数

    result_mq_list = []
    process_list = []
    for i in range(cpu_nums):
        # 每个子进程处理一部分数据
        step = int(item_counts / cpu_nums) + 1
        start_idx = step * i
        end_idx = step * (i + 1)
        if end_idx >= item_counts:
            end_idx = item_counts
        sub_items = all_items[start_idx:end_idx]

        # 进程通信 Queue
        result_mq = multiprocessing.Queue()
        process = multiprocessing.Process(target=item_similarity_multi_process_func2,
                                          args=(co_items, num_items, sub_items, result_mq))
        process.start()
        process_list.append(process)
        result_mq_list.append(result_mq)

    # 等待子进程结束

    # 所有进程均执行完毕
    # 获取数据
    for queue in result_mq_list:
        sub_item_sims = queue.get()  # dict 2d
        # 合并dict  2d
        if item_sims is None:
            item_sims = sub_item_sims
        else:
            item_sims.update(sub_item_sims)

    return item_sims


def item_similarity_multi_thread(_train, iuf=False):
    """
    多线程计算物品之间的相似度
    :param _train: dict{list} 训练集
    :param iuf: bool 是否考虑用户活跃度
    :return:
    """

    # 多线程计算物品之间的相似度
    class SimilarityThread(threading.Thread):
        def __init__(self, _item_sims: dict, _num_items: dict, _item: int, _related_items: dict):
            threading.Thread.__init__(self)
            self.__item_sims = _item_sims
            self.__num_items = _num_items
            self.__item = _item
            self.__related_items = _related_items

        def run(self):
            # i物品与其他物品
            for item_jj, cij in self.__related_items.items():
                # j物品 cij表示i j被多少用户共有
                if self.__item not in self.__item_sims:
                    self.__item_sims[self.__item] = dict()
                self.__item_sims[self.__item][item_jj] = \
                    cij / math.sqrt(self.__num_items[self.__item] * self.__num_items[item_jj])

    # 统计矩阵 C:表示两个物品被多少用户共同交互过 N表示每个用品被多少用户交互过
    item_sims = dict()
    co_items = dict()
    num_items = dict()

    for u, items in _train.items():
        if iuf:
            iuf_value = 1 / math.log(1 + len(items) * 1.0)  # IUF 该用户对物品相似度的贡献
        else:
            iuf_value = 1
        for item_i in items:
            if item_i not in co_items:
                co_items[item_i] = dict()
            if item_i not in num_items:
                num_items[item_i] = 0
            num_items[item_i] += 1

            for item_j in items:
                if item_i == item_j:
                    continue
                if item_j not in co_items[item_i]:
                    co_items[item_i][item_j] = 0
                co_items[item_i][item_j] += iuf_value

    items = list(co_items.keys())
    for ii in range(len(items)):
        item = items[ii]
        thread = SimilarityThread(item_sims, num_items, item, co_items.get(item))
        thread.start()
        thread.join()

    return item_sims


def item_similarity_single_thread(_train, iuf=False):
    """
    单进程 单线程常规计算相似度
    :param _train:
    :param iuf: bool 是否考虑用户活跃度
    :return:
    """
    # 统计矩阵 C:表示两个物品被多少用户共同交互过 N表示每个用品被多少用户交互过
    item_sims = dict()
    co_items = dict()
    num_items = dict()

    for user, items in _train.items():
        if iuf:
            iuf_value = 1 / math.log(1 + len(items) * 1.0)  # IUF 该用户对物品相似度的贡献
        else:
            iuf_value = 1
        for item_i in items:
            if item_i not in co_items:
                co_items[item_i] = dict()
            if item_i not in num_items:
                num_items[item_i] = 0
            num_items[item_i] += 1

            for item_j in items:
                if item_i == item_j:
                    continue
                if item_j not in co_items[item_i]:
                    co_items[item_i][item_j] = 0
                co_items[item_i][item_j] += iuf_value

    # 计算物品之间的相似度
    for item_i, related_items in co_items.items():
        if item_i not in item_sims:
            item_sims[item_i] = dict()
        # i物品与其他物品
        for item_j, cij in related_items.items():
            # j物品 cij表示i j被多少用户共有
            item_sims[item_i][item_j] = cij / math.sqrt(num_items[item_i] * num_items[item_j])
    return item_sims


def item_similarity_iuf(_train):
    """
    Item-IUF 相似度计算的改进
    Inverse User Frequency 用户活跃度对数的倒数的参数
    活跃用户对物品相似度的贡献应当小于不活跃用户
    :param _train: dict 2d
    :return:
    """
    # item_sims = item_similarity_multi_thread(_train, True)
    item_sims = item_similarity_multi_process(_train, True)
    # item_sims = item_similarity_single_thread(_train, True)
    return item_sims


def item_similarity_normal(_item_sims):
    """
    对物品相似度矩阵最大值归一化
    :param _item_sims:
    :return:
    """
    for item_i, related_items in _item_sims.items():
        # i物品与其他物品的相似度
        sims = [sim_ij for (_, sim_ij) in related_items.items()]
        max_sim = max(sims)

        for item_j, sim_ij in related_items.items():
            _item_sims[item_i][item_j] /= max_sim


def compare_item_similarity(_item_sims_1, _item_sims_2):
    """
    比较两种不同方式的得到的相似度矩阵是否存在差异
    :param _item_sims_1:
    :param _item_sims_2:
    :return:
    """
    # key的差异
    diff = set(_item_sims_1.keys()) - set(_item_sims_2.keys())
    print(diff)
    # value的差异
    for key in _item_sims_1.keys():
        val1 = _item_sims_1.get(key)
        val2 = _item_sims_2.get(key)
        for key_2 in val1.keys():
            vall1 = val1.get(key_2)
            vall2 = val2.get(key_2)
            if vall1 != vall2:
                print(key, key_2, vall1, vall2, "diff value")


def item_similarity(train_data):
    """
    计算物品之间的相似度
    :param train_data:
    :return:
    """
    start = time.time()
    # 多进程 多线程 单线程 iuf
    # item_sims = item_similarity_multi_thread(train_data)
    # item_sims = item_similarity_multi_process(train_data)
    item_sims = item_similarity_single_thread(train_data)
    # item_sims = item_similarity_iuf(train_data)

    # 归一化
    item_similarity_normal(item_sims)
    print("item similarity done, cost " + str(time.time() - start) + " s")
    return item_sims


def recommend_multi_process_func1(_users, _train, _item_sims, _nearest_k, _top_n, _result_mq):
    _recommend_lists = dict()
    for user in _users:
        # 该用户交互过的物品
        interacted_items = _train.get(user)
        rank = dict()
        for item_i, rating in interacted_items.items():
            # 根据相似度矩阵, 找到距离物品i最近的K个物品
            nearest_items = sorted(_item_sims[item_i].items(), key=operator.itemgetter(1), reverse=True)[0:_nearest_k]
            for item_j, wj in nearest_items:
                if item_j in interacted_items:
                    continue  # 该物品已经被用户交互过
                if item_j not in rank:
                    rank[item_j] = 0
                rank[item_j] += rating * wj

        # 选择Top N 作为该用户的推荐
        top_n_items = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:_top_n]
        # 转成list
        _recommend_lists[user] = [item for (item, _) in top_n_items]
    _result_mq.put(_recommend_lists)


def recommend_multi_process(_train, _item_sims, _nearest_k, _top_n):
    """
    多进程推荐
    :param _train: dict{list} 训练集
    :param _item_sims: dict 2d 物品相似度矩阵
    :param _nearest_k: int 使用最近的k个物品计算评分
    :param _top_n: int 返回评分最高的n个物品
    :return recommend_lists: dict{list} 每个用户的top n推荐列表
    """

    recommend_lists = dict()

    cpu_nums = multiprocessing.cpu_count()
    users = [user for user in _train]  # 所有用户集合
    user_counts = len(users)

    result_mq_list = []
    process_list = []
    for i in range(cpu_nums):
        # 每个子进程处理一部分数据
        step = int(user_counts / cpu_nums) + 1
        start_idx = step * i
        end_idx = step * (i + 1)
        if end_idx >= user_counts:
            end_idx = user_counts
        sub_users = users[start_idx:end_idx]

        # 进程通信 Queue
        result_mq = multiprocessing.Queue()
        process = multiprocessing.Process(target=recommend_multi_process_func1,
                                          args=(sub_users, _train, _item_sims, _nearest_k, _top_n, result_mq))

        process.start()
        process_list.append(process)
        result_mq_list.append(result_mq)

    # 等待子进程结束
    for process in process_list:
        process.join()
    # 所有进程均执行完毕

    # 获取数据
    for queue in result_mq_list:
        recommend_lists.update(queue.get())

    return recommend_lists


def recommend_multi_thread(_train, _item_sims, _nearest_k, _top_n):
    """
    多线程推荐
    :param _train: dict{list} 训练集
    :param _item_sims: dict 2d 物品相似度矩阵
    :param _nearest_k: int 使用最近的k个物品计算评分
    :param _top_n: int 返回评分最高的n个物品
    :return recommend_lists: dict{list} 每个用户的top n推荐列表
    """

    # 自定义多线程类 继承自threading.Thread
    class RecommendThread(threading.Thread):
        def __init__(self, _user, _interact_items, _item_sims, _nearest_k, _top_n, _recommends):
            threading.Thread.__init__(self)
            self.__user = _user
            self.__interact_items = _interact_items
            self.__item_sims = _item_sims
            self.__nearest_k = _nearest_k
            self.__top_n = _top_n
            self.__recommends = _recommends

        def run(self):
            rank = dict()
            # 该用户交互过的物品
            for item_i, rating in self.__interact_items.items():
                # 根据相似度矩阵, 找到距离物品i最近的K个物品
                nearest_items = sorted(self.__item_sims[item_i].items(), key=operator.itemgetter(1), reverse=True)[
                                0:self.__nearest_k]
                for item_j, wj in nearest_items:
                    if item_j in self.__interact_items:
                        continue  # 该物品已经被用户交互过
                    if item_j not in rank:
                        rank[item_j] = 0
                        rank[item_j] += rating * wj

            # 根据预测评分 排序选择Top N 作为该用户的推荐
            top_n_rank = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:self.__top_n]
            top_n_rank_list = [item for (item, _) in top_n_rank]
            self.__recommends[self.__user] = top_n_rank_list

    recommend_lists = dict()
    users = list(_train.keys())
    for ii in range(len(users)):
        user = users[ii]
        thread1 = RecommendThread(user, _train.get(user), _item_sims, _nearest_k, _top_n, recommend_lists)
        thread1.start()
        thread1.join()
    return recommend_lists


def recommend_single_thread(_train, _item_sims, _nearest_k, _top_n):
    """
    单线程推荐
    :param _train: dict{list} 训练集
    :param _item_sims: dict 2d 物品相似度矩阵
    :param _nearest_k: int 使用最近的k个物品计算评分
    :param _top_n: int 返回评分最高的n个物品
    :return recommend_lists: dict{list} 每个用户的top n推荐列表
    """

    recommend_lists = dict()

    for user, interacted_items in _train.items():
        rank = dict()
        # 该用户交互过的物品
        for item_i, rating in interacted_items.items():
            # 根据相似度矩阵, 找到距离物品i最近的K个物品
            nearest_items = sorted(_item_sims[item_i].items(), key=operator.itemgetter(1), reverse=True)[0:_nearest_k]
            for item_j, wj in nearest_items:
                if item_j in interacted_items:
                    continue  # 该物品已经被用户交互过
                if item_j not in rank:
                    rank[item_j] = 0
                rank[item_j] += rating * wj  # rating是用户对物品i的评分

        # 选择Top N 作为该用户的推荐
        top_n_items = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:_top_n]
        # 转成list
        recommend_lists[user] = [item for (item, _) in top_n_items]
    return recommend_lists


def recommend(train_data, item_sims, nearest_k=5, top_n=10):
    """
    执行推荐 为每个用户都推荐物品
    :param train_data: dict{list} 训练集
    :param item_sims: dict 2d 物品相似度矩阵
    :param nearest_k: int 使用最近的k个物品计算评分
    :param top_n: int 返回评分最高的n个物品
    :return recommend_lists: dict{list} 每个用户的top n推荐列表
    """
    start = time.time()
    # recommend_list = recommend_multi_thread(train_data, item_sims, nearest_k, top_n)
    recommend_list = recommend_multi_process(train_data, item_sims, nearest_k, top_n)
    # recommend_list = recommend_single_thread(train_data, item_sims, nearest_k, top_n)
    print("recommend done, cost " + str(time.time() - start) + " s")
    return recommend_list


if __name__ == '__main__':
    start_time = time.time()

    train, test = utils.split_data(utils.load_data("./data/ratings.dat"), 8, 1)

    W = item_similarity(train)

    recommends = recommend(train, W, nearest_k=10, top_n=10)

    p = utils.precision(train, test, recommends)

    r = utils.recall(train, test, recommends)

    c = utils.coverage(train, recommends)

    po = utils.popularity(train, recommends)

    cost_time = time.time() - start_time

    print(p, r, c, po)
    print("cost time " + str(cost_time) + " s")
