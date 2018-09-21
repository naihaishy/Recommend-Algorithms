# -*- coding:utf-8 -*-
# @Time : 2018/9/19 9:13
# @Author : naihai

import math
import time
import random
import operator
import threading
import multiprocessing

"""
基于物品的协同过滤算法
最简单的ItemCF
"""


def item_similarity_multi_process_func1(_co_items, _num_items):
    """
    :param _co_items: dict 2d 共现矩阵 表示每两个物品被共有的统计矩阵
    :param _num_items: dict 每个物品被多少用户拥有
    :return:
    """
    for user, items in train.items():
        for item_i in items:
            if item_i not in _num_items:
                _num_items[item_i] = 0
            _num_items[item_i] += 1
            for item_j in items:
                if item_i == item_j:
                    continue
                if item_j not in _co_items[item_i]:
                    item_i_dict = _co_items[item_i]
                    item_i_dict[item_j] = 0
                    _co_items[item_i] = item_i_dict
                item_i_dict = _co_items[item_i]
                item_i_dict[item_j] += 1
                _co_items[item_i] = item_i_dict


def item_similarity_multi_process_func2(_item_sims, _co_items, _num_items, _items):
    """
    :param _item_sims: dict 2d 物品相似度矩阵
    :param _co_items: dict 2d 共现矩阵 表示每两个物品被共有的统计矩阵
    :param _num_items: dict 每个物品被多少用户拥有
    :param _items: list 物品集合
    :return:
    """
    for item_i in _items:
        related_items = _co_items.get(item_i)
        # i物品与其他物品
        for item_j, cij in related_items.items():
            # j物品 cij表示物品i和物品j被多少用户共有
            if item_j not in _item_sims[item_i]:
                item_i_dict = _item_sims[item_i]
                item_i_dict[item_j] = 0
                W[item_i] = item_i_dict

            item_i_dict = W[item_i]
            item_i_dict[item_j] = cij / math.sqrt(_num_items[item_i] * _num_items[item_j])
            _item_sims[item_i] = item_i_dict


def item_similarity_multi_process(_train):
    """
    多进程计算物品之间的相似度
    :param _train: dict{list} 训练集
    :return:
    """
    cpus = multiprocessing.cpu_count()
    # 统计矩阵 co_items:表示两个物品被多少用户共同交互过 num_items表示每个用品被多少用户交互过
    manager = multiprocessing.Manager()
    item_sims = manager.dict()
    co_items = manager.dict()
    num_items = manager.dict()

    # 初始化二维dict 必须在该进程中初始化
    for user, items in _train.items():
        for item in items:
            if item not in co_items:
                co_items[item] = manager.dict()
    # 多进程
    p1 = multiprocessing.Process(target=item_similarity_multi_process_func1, args=(co_items, num_items))
    p1.start()
    p1.join()

    # 计算物品之间的相似度
    all_items = list(co_items.keys())
    item_counts = len(all_items)  # 物品总数
    # 初始化二维dict
    for item in all_items:
        if item not in W:
            item_sims[item] = manager.dict()

    # 使用进程池
    pool = multiprocessing.Pool(processes=cpus)
    num_step = cpus * 10
    for i in range(num_step):
        step = int(item_counts / num_step) + 1
        start_idx = step * i
        end_idx = step * (i + 1)
        if end_idx >= item_counts:
            end_idx = item_counts
        pool.apply(item_similarity_multi_process_func2,
                   args=(item_sims, co_items, num_items, all_items[start_idx:end_idx]))
    pool.close()
    pool.join()
    return W


def item_similarity_multi_thread(_train):
    """
    多线程计算物品之间的相似度
    :param _train: dict{list} 训练集
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
                if item_jj not in self.__item_sims[self.__item]:
                    self.__item_sims[self.__item][item_jj] = 0
                self.__item_sims[self.__item][item_jj] = \
                    cij / math.sqrt(self.__num_items[self.__item] * self.__num_items[item_jj])

    # 统计矩阵 C:表示两个物品被多少用户共同交互过 N表示每个用品被多少用户交互过
    item_sims = dict()
    co_items = dict()
    num_items = dict()

    for u, items in _train.items():
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
                co_items[item_i][item_j] += 1

    items = list(co_items.keys())
    for i in range(len(items)):
        item = items[i]
        thread = SimilarityThread(item_sims, num_items, item, co_items.get(item))
        thread.start()
        thread.join()

    return item_sims


def item_similarity_single_thread(_train):
    """
    单进程 单线程常规计算相似度
    :param _train:
    :return:
    """
    # 统计矩阵 C:表示两个物品被多少用户共同交互过 N表示每个用品被多少用户交互过
    item_sims = dict()
    co_items = dict()
    num_items = dict()

    for user, items in train.items():
        for item_i in items:
            if item_i not in num_items:
                num_items[item_i] = 0
            num_items[item_i] += 1

            for item_j in items:
                if item_i == item_j:
                    continue
                if item_i not in co_items:
                    co_items[item_i] = dict()
                if item_j not in co_items[item_i]:
                    co_items[item_i][item_j] = 0
                co_items[item_i][item_j] += 1

    # 计算物品之间的相似度
    for item_i, related_items in co_items.items():
        # i物品与其他物品
        for item_j, cij in related_items.items():
            # j物品 cij表示i j被多少用户共有
            if item_i not in item_sims:
                item_sims[item_i] = dict()
            if item_j not in item_sims[item_i]:
                item_sims[item_i][item_j] = 0
            item_sims[item_i][item_j] = cij / math.sqrt(num_items[item_i] * num_items[item_j])
    return item_sims


def item_similarity_iuf(_train):
    """
    Item-IUF 相似度计算的改进
    Inverse User Frequency 用户活跃度对数的倒数的参数
    活跃用户对物品相似度的贡献应当小于不活跃用户
    :param _train:
    :return:
    """
    # 统计矩阵 C:表示两个物品被多少用户共同交互过 N表示每个用品被多少用户交互过
    item_sims = dict()
    co_items = dict()
    num_items = dict()

    for user, items in train.items():
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
                co_items[item_i][item_j] += 1 / math.log(1 + len(items) * 1.0)

    # 计算物品之间的相似度
    for item_i, related_items in co_items.items():
        # i物品与其他物品
        for item_j, cij in related_items.items():
            # j物品 cij表示i j被多少用户共有
            if item_i not in item_sims:
                item_sims[item_i] = dict()
            if item_j not in item_sims[item_i]:
                item_sims[item_i][item_j] = 0
            item_sims[item_i][item_j] = cij / math.sqrt(num_items[item_i] * num_items[item_j])
    return item_sims


def item_similarity(train_data):
    """
    计算物品之间的相似度
    :param train_data:
    :return:
    """
    start = time.time()

    # W1 = item_similarity_multi_thread(train_data)
    # W2 = item_similarity_multi_process(train_data)
    # item_sims = item_similarity_single_thread(train_data)
    item_sims = item_similarity_iuf(train_data)
    print("item similarity done, cost " + str(time.time() - start) + " s")
    return item_sims


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
            for item_i in self.__interact_items:
                # 根据相似度矩阵, 找到距离物品i最近的K个物品
                nearest_items = sorted(self.__item_sims[item_i].items(), key=operator.itemgetter(1), reverse=True)[
                                0:self.__nearest_k]
                for item_j, wj in nearest_items:
                    if item_j in self.__interact_items:
                        continue  # 该物品已经被用户交互过
                    if item_j not in rank:
                        rank[item_j] = 0
                        rank[item_j] += 1 * wj  # ri=1 是用户对物品i的评分 rating

            # 根据预测评分 排序选择Top N 作为该用户的推荐
            top_n_rank = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:self.__top_n]
            top_n_rank_list = [item for (item, _) in top_n_rank]
            self.__recommends[self.__user] = top_n_rank_list

    recommend_lists = dict()
    users = list(_train.keys())
    for i in range(len(users)):
        user = users[i]
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
        for item_i in interacted_items:
            # 根据相似度矩阵, 找到距离物品i最近的K个物品
            nearest_items = sorted(_item_sims[item_i].items(), key=operator.itemgetter(1), reverse=True)[0:_nearest_k]
            for item_j, wj in nearest_items:
                if item_j in interacted_items:
                    continue  # 该物品已经被用户交互过
                if item_j not in rank:
                    rank[item_j] = 0
                rank[item_j] += 1 * wj  # ri=1 是用户对物品i的评分 rating

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
    recommend_list = recommend_single_thread(train_data, item_sims, nearest_k, top_n)
    print("recommend done, cost " + str(time.time() - start) + " s")
    return recommend_list


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


def split_data(all_data, m, k, seed):
    """
    切分数据集 留一法
    :param all_data:
    :param m:
    :param k:
    :param seed:
    :return:
    """
    start = time.time()
    train_data = dict()
    test_data = dict()
    random.seed(seed)

    for user, items in all_data.items():
        # 初始化dict
        if train_data.get(user) is None:
            train_data[user] = list()
        if test_data.get(user) is None:
            test_data[user] = list()

        for item in items:
            if random.randint(0, m) == k:
                test_data[user].append(item)
            else:
                train_data[user].append(item)
    print("split data done, cost " + str(time.time() - start) + " s")
    return train_data, test_data


def load_data(file):
    """
    加载数据
    :param file:
    :return:
    """
    start = time.time()
    all_data = dict()
    with open(file, 'r') as f:
        line = f.readline()
        while line is not None and line != '':
            arr = line.split("::")
            user = int(arr[0])
            item = int(arr[1])
            if user not in all_data:
                all_data[user] = list()
            all_data[user].append(item)
            line = f.readline()
    print("load data done, cost " + str(time.time() - start) + " s")
    return all_data
    return_data = dict()
    for key in all_data.keys():
        if key <= 1000:
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
        f.write(str(_nearest_k) + ", " + str(_top_n) + ", " + str(_p) + ", " + str(_r) \
                + ", " + str(_c) + ", " + str(_po) + "\n")


def plot_result():
    pass


if __name__ == '__main__':
    start_time = time.time()
    data = load_data("./data/ratings.dat")

    train, test = split_data(data, 8, 1, time.time())

    W = item_similarity(train)

    nearest_k = 10
    top_n = 10
    recommends = recommend(train, W, nearest_k, top_n)

    p = precision(train, test, recommends)

    r = recall(train, test, recommends)

    c = coverage(train, recommends)

    po = popularity(train, recommends)

    print(p, r, c, po)

    cost_time = time.time() - start_time

    print("cost time " + str(cost_time) + " s")

    exit(0)

    for i in range(5, 100, 5):
        nearest_k = i
        top_n = 10
        recommends = recommend(train, W, nearest_k, top_n)

        p = precision(train, test, recommends)

        r = recall(train, test, recommends)

        c = coverage(train, recommends)

        po = popularity(train, recommends)

        print(p, r, c, po)
        cost_time = time.time() - start_time

        save_result(cost_time, nearest_k, top_n, p, r, c, po)

        print("cost time " + str(cost_time) + " s")
