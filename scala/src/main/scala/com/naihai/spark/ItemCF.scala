package com.naihai.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel


object ItemCF {


  /**
    * 计算物品的相似度
    *
    * @param train_data 训练集
    * @return
    */
  def item_similarity(train_data: RDD[(Int, Int, Int)]): RDD[(Int, Int, Double)] = {
    // 去除评分信息
    val user_rdd = train_data.map(r => (r._1, r._2)) // (user, item)

    val user_rdd2 = user_rdd
      .join(user_rdd) // (user,(item i,item j)) 表示物品ij均被该用户交互过
      .map(r => (r._2, 1)) // ((item i ,item j),1)
      .reduceByKey(_ + _) // ((item i,item j), count) 表示两个物品被多少用户共同交互过
    user_rdd2.persist(StorageLevel.MEMORY_ONLY_SER)

    // 统计每个物品被交互用户数
    val user_rdd3 = user_rdd2
      .filter(r => r._1._1 == r._1._2) // ((item i, item i), count)
      .map(r => (r._1._1, r._2)) // (item i, N(i)) N(i)表示物品i被多少用户交互过
    user_rdd3.persist(StorageLevel.MEMORY_ONLY_SER)


    val user_rdd4 = user_rdd2
      .filter(r => r._1._1 != r._1._2) // ((item i,item j), count) C(i,j) 共现矩阵 i!=j
      .map(r => (r._1._1, (r._1._1, r._1._2, r._2))) // (item i, (item i, item j, count))
      .join(user_rdd3) // (item i, ((item i, item j, count), N(i)))
      .map(r => (r._2._1._2, (r._2._1._1, r._2._1._2, r._2._1._3, r._2._2))) // (item j, (item i, item j, count, N(i)))
    user_rdd4.persist(StorageLevel.MEMORY_ONLY_SER)
    user_rdd2.unpersist()

    val user_rdd5 = user_rdd4
      .join(user_rdd3) // (item j, ((item i, item j, count, N(i)), N(j)))
      .map(r => (r._2._1._1, r._2._1._2, r._2._1._3, r._2._1._4, r._2._2)) // (item i,item j, count, N(i), N(j))
    user_rdd5.persist(StorageLevel.MEMORY_ONLY_SER)

    // 计算同现相似度
    user_rdd5.map(r => (r._1, r._2, r._3 / math.sqrt(r._4 * r._5))) // (item i,item j, count/sqrt(N(i) * N(j)) )
  }


  /**
    * 生成推荐列表
    *
    * @param train_data 训练集
    * @param similarity 物品相似度矩阵
    * @param nearest_k  选择最近邻k个物品计算评分
    * @param top_n      推荐n个评分最高的物品
    * @return
    */
  def recommend(train_data: RDD[(Int, Int, Int)], similarity: RDD[(Int, Int, Double)], nearest_k: Int, top_n: Int): RDD[(Int, List[(Int, Double)])] = {

    val rdd1 = similarity
      .map(r => (r._1, (r._2, r._3)))
      .groupByKey()
      .map(r => (r._1, r._2.toArray.sortBy(_._2).reverse.take(nearest_k))) // (item i, Array((item, sim), ...))
      .flatMap(r => for (w <- r._2) yield (r._1, (w._1, w._2))) //(item i, (item j, sim))

    val rdd2 = train_data.map(r => (r._1, (r._2, r._3))) // (user, (item, rating))

    val rdd3 = train_data
      .map(r => (r._1, r._2)) // (user, item)
      .groupByKey() // (user, Array) Array为该用户交互过的物品集合
      .join(rdd2) // (user, (Array, (item, rating)))
      .map(r => (r._2._2._1, (r._1, r._2._2._2, r._2._1))) // (item, (user, rating , Array))
      .join(rdd1) // (item , ((user, rating , Array), (item j, sim) ))
      .filter(r => !r._2._1._3.toSet.contains(r._2._2._1)) // item j不在该用户已经交互过的物品集中
      .map(r => (r._2._1._1, r._1, r._2._1._2, r._2._2._1, r._2._2._2)) // (user , item i, rating, item j, sim)

    val rdd4 = rdd3
      .map(r => ((r._1, r._4), r._3 * r._5)) // ((user , item j), sim*rating)
      .reduceByKey(_ + _) // ((user, item j ), preference)
      .map(r => (r._1._1, (r._1._2, r._2))) // (user, (item , preference))
      .groupByKey()

    rdd4.map(r => (r._1, r._2.toArray.sortBy(_._2).reverse.take(top_n).toList)) //根据评分进行排序 降序 选择前top_n

  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("ItemCF").master("spark://Future:7077").getOrCreate()

    val train_data = Utils.load_data(spark, "/home/hadoop/Runs/data/train.dat", "::")
    val test_data = Utils.load_data(spark, "/home/hadoop/Runs/data/test.dat", "::")

    val item_sims = item_similarity(train_data)
    val recommend_list = recommend(train_data, item_sims, 10, 10)
    recommend_list.saveAsTextFile("/home/hadoop/Runs/data/rrr")

    val precision = Utils.precision(test_data, recommend_list)
    val recall = Utils.recall(test_data, recommend_list)
    val coverage = Utils.coverage(train_data, recommend_list)
    val popularity = Utils.popularity(train_data, recommend_list)

    println(precision, recall, coverage, popularity)
    spark.stop()
  }

}
