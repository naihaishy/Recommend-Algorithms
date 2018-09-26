package com.naihai.study

import scala.collection.mutable
import scala.io.Source
import scala.util.Random


object Utils {


  /**
    * 加载数据
    */
  def load_data(filename: String): mutable.HashMap[Int, mutable.HashMap[Int, Int]] = {
    val start = System.currentTimeMillis()
    val data: mutable.HashMap[Int, mutable.HashMap[Int, Int]] = mutable.HashMap()
    Source.fromFile(filename).getLines().foreach(line => {
      val arr = line.split("::")
      val user = arr(0).toInt
      val item = arr(1).toInt
      val rating = arr(2).toInt
      if (!data.contains(user))
        data.put(user, mutable.HashMap())
      data(user).put(item, rating)
    })
    println("load data done, cost " + (System.currentTimeMillis() - start) + " ms")
    data
  }

  /**
    * 切分数据集
    * 切为M分 k分作为测试集 剩下的作为训练集
    */
  def split_data(data: mutable.HashMap[Int, mutable.HashMap[Int, Int]], m: Int, k: Int):
  (mutable.HashMap[Int, mutable.HashMap[Int, Int]], mutable.HashMap[Int, mutable.HashMap[Int, Int]]) = {
    val start = System.currentTimeMillis()
    val train_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]] = mutable.HashMap()
    val test_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]] = mutable.HashMap()
    data.foreach(elem => {
      val test_data_keys = Random.shuffle(elem._2.keys.toList).take(elem._2.size / m * k).toSet
      // 一个用户的训练集与测试集
      val train_user_data: mutable.HashMap[Int, Int] = mutable.HashMap()
      val test_user_data: mutable.HashMap[Int, Int] = mutable.HashMap()
      elem._2.foreach(item => {
        if (test_data_keys.contains(item._1))
          test_user_data.put(item._1, item._2)
        else
          train_user_data.put(item._1, item._2)
      })

      train_data.put(elem._1, train_user_data)
      test_data.put(elem._1, test_user_data)
    })

    println("split data done, cost " + (System.currentTimeMillis() - start) + " ms")
    (train_data, test_data)
  }

  /**
    * 计算precision准确度
    */
  def precision(train_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]],
                test_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]],
                recommend_list: mutable.HashMap[Int, List[Int]]): Double = {

    val start = System.currentTimeMillis()
    var hit_num = 0
    var all_num = 0

    train_data.foreach(elem => {
      // elem._1 user elem._2 该用户交互的物品集合
      val tu = test_data(elem._1).keys.toSet //测试集中该用户交互的物品
      val rank = recommend_list(elem._1).toSet //给该用户推荐的物品
      hit_num += (tu & rank).size //两个集合取交集
      all_num += rank.size
    })

    println("calculate precision done, cost " + (System.currentTimeMillis() - start) + " ms")
    hit_num / (all_num * 1.0)
  }

  /**
    * 计算recall召回率
    */
  def recall(train_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]],
             test_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]],
             recommend_list: mutable.HashMap[Int, List[Int]]): Double = {

    val start = System.currentTimeMillis()
    var hit_num = 0
    var all_num = 0

    train_data.foreach(elem => {
      // elem._1 user elem._2 该用户交互的物品集合
      val tu = test_data(elem._1).keys.toSet //测试集中该用户交互的物品
      val rank = recommend_list(elem._1).toSet //给该用户推荐的物品
      hit_num += (tu & rank).size //两个集合取交集
      all_num += tu.size
    })

    println("calculate recall done, cost " + (System.currentTimeMillis() - start) + " ms")
    hit_num / (all_num * 1.0)
  }

  /**
    * 计算coverage覆盖率
    */
  def coverage(train_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]],
               recommend_list: mutable.HashMap[Int, List[Int]]): Double = {

    val start = System.currentTimeMillis()
    var recommend_items: Set[Int] = Set() //所有的推荐物品集合
    var all_items: Set[Int] = Set() //所有物品集合
    train_data.foreach(elem => {
      // elem._1 user elem._2 该用户交互的物品集合
      val rank = recommend_list(elem._1).toSet //给该用户推荐的物品
      recommend_items = recommend_items.union(rank)
      all_items = all_items.union(elem._2.keys.toSet)
    })
    println("calculate coverage done, cost " + (System.currentTimeMillis() - start) + " ms")
    recommend_items.size / (all_items.size * 1.0)

  }

  /**
    * 计算popularity流行度
    */
  def popularity(train_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]],
                 recommend_list: mutable.HashMap[Int, List[Int]]): Double = {

    val start = System.currentTimeMillis()
    val item_popularity: mutable.HashMap[Int, Int] = mutable.HashMap() //所有物品的流行度
    train_data.foreach(elem => {
      // elem._1 user elem._2 该用户交互的物品集合
      elem._2.foreach(item => {
        if (!item_popularity.contains(item._1))
          item_popularity.put(item._1, 0)
        item_popularity(item._1) += 1
      })
    })

    // 计算推荐项目的平均流行度
    var ret = 0.0
    var count = 0
    train_data.foreach(elem => {
      // elem._1 user elem._2 该用户交互的物品集合
      val rank = recommend_list(elem._1) //给该用户推荐的物品集合
      rank.foreach(item => {
        ret += math.log(1 + item_popularity(item))
      })
      count += rank.size
    })
    println("calculate popularity done, cost " + (System.currentTimeMillis() - start) + " ms")
    ret / (count * 1.0)
  }


}
