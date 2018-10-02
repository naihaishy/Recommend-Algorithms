package com.naihai.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

import scala.util.Random

object Utils {


  /**
    * 加载数据
    *
    * @param spark SparkSession
    * @param file  输入数据路径
    * @param seq   数据分割符号
    * @return
    */
  def load_data(spark: SparkSession, file: String, seq: String): RDD[(Int, Int, Int)] = {
    spark.sparkContext.textFile(file).map(line => {
      val fields = line.split(seq)
      (fields(0).toInt, fields(1).toInt, fields(2).toInt)
    })
  }


  /**
    * 切分数据集
    *
    * @param data 输入数据
    * @param m    切为M分
    * @param k    k分作为测试集  剩下的作为训练集
    * @return
    */
  def split_data(data: RDD[(Int, Int, Int)], m: Int, k: Int): RDD[(Int, Int, Int)] = {

    ???

  }


  /**
    * 计算precision准确度
    *
    * @param test_data      测试集
    * @param recommend_list 所有用户的推荐top_n 推荐列表
    * @return
    */
  def precision(test_data: RDD[(Int, Int, Int)], recommend_list: RDD[(Int, List[(Int, Double)])]): Double = {
    val rdd1 = test_data.map(r => (r._1, r._2)).groupByKey().map(r => (r._1, r._2.toList))
    val rdd2 = recommend_list.map(r => (r._1, r._2.map(_._1)))
    val rdd3 = rdd1
      .join(rdd2) // (user, (List, List))
      .map(r => (r._1, (r._2._1.toSet.intersect(r._2._2.toSet).size, r._2._2.size))) // (user, (hit_num, all_num))
      .map(r => (r._2._1, r._2._2)) // (hit_num, all_num)
      .reduce((x, y) => (x._1 + y._1, x._2 + y._2)) // (hit_nums, all_nums)
    rdd3._1 / (rdd3._2 * 1.0)
  }

  /**
    * 计算recall召回率
    *
    * @param test_data      测试集
    * @param recommend_list 所有用户的推荐top_n 推荐列表
    * @return
    */
  def recall(test_data: RDD[(Int, Int, Int)], recommend_list: RDD[(Int, List[(Int, Double)])]): Double = {
    val rdd1 = test_data.map(r => (r._1, r._2)).groupByKey().map(r => (r._1, r._2.toList))
    val rdd2 = recommend_list.map(r => (r._1, r._2.map(_._1)))
    val rdd3 = rdd1
      .join(rdd2) // (user, (List, List))
      .map(r => (r._1, (r._2._1.toSet.intersect(r._2._2.toSet).size, r._2._1.size))) // (user, (hit_num, all_num))
      .map(r => (r._2._1, r._2._2)) // (hit_num, all_num)
      .reduce((x, y) => (x._1 + y._1, x._2 + y._2)) // (hit_nums, all_nums)
    rdd3._1 / (rdd3._2 * 1.0)
  }


  /**
    * 计算coverage覆盖率
    *
    * @param train_data     训练集
    * @param recommend_list 所有用户的推荐top_n 推荐列表
    * @return
    */
  def coverage(train_data: RDD[(Int, Int, Int)], recommend_list: RDD[(Int, List[(Int, Double)])]): Double = {
    // 推荐的所有物品集合
    val recommend_items = recommend_list.map(r => r._2.map(_._1).toSet).reduce((x, y) => x.union(y))
    val all_items = train_data.map(r => (r._1, r._2)).groupByKey().map(r => r._2.toSet).reduce((x, y) => x.union(y))
    recommend_items.size / (all_items.size * 1.0)
  }

  /**
    * 计算popularity流行度
    *
    * @param train_data     训练集
    * @param recommend_list 所有用户的推荐top_n 推荐列表
    * @return
    */
  def popularity(train_data: RDD[(Int, Int, Int)], recommend_list: RDD[(Int, List[(Int, Double)])]): Double = {
    // 统计项目的流行度
    val item_popularity = train_data.map(r => (r._2, 1)).reduceByKey(_ + _)

    // 计算推荐项目的平均流行度
    val all_popularity = recommend_list
      .flatMap(r => for (w <- r._2) yield (r._1, w._1))
      .map(r => (r._2, r._1)) //(item, user)
      .join(item_popularity) // (item, (user, popularity))
      .map(r => math.log(1 + r._2._2)) // popularity
      .sum() //
    val all_items = recommend_list.map(_._2.size).sum() // 所有用户获得的推荐节目总数
    all_popularity / (all_items * 1.0)
  }


}
