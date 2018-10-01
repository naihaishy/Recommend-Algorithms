package com.naihai.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

import scala.util.Random

object ItemCF {

  /**
    * 加载数据
    *
    * @param spark SparkSession
    * @param file  输入数据路径
    * @param seq   数据分割符号
    * @return
    */
  def load_data(spark: SparkSession, file: String, seq: String): RDD[(Int, Int, Int)] = {
    val rdd = spark.sparkContext.textFile(file)
    val ret = rdd.map(line => {
      val fields = line.split(seq);
      (fields(0).toInt, fields(1).toInt, fields(2).toInt)
    })
    ret
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
    * 计算物品的相似度
    *
    * @param train_data 训练集
    * @return
    */
  def item_similarity(train_data: RDD[(Int, Int, Int)]): RDD[(Int, Int, Double)] = {
    // 去除评分信息
    val user_rdd = train_data.map(r => (r._1, r._2)) // (user, item)

    val user_rdd3 = user_rdd
      .join(user_rdd) // (user,(item i,item j)) 表示物品ij均被该用户交互过
      .map(r => (r._2, 1)) // ((item i ,item j),1)
      .reduceByKey((x, y) => x + y) // ((item i,item j), count) 表示两个物品被多少用户共同交互过

    user_rdd3.persist(StorageLevel.MEMORY_ONLY_SER)

    // 统计每个物品被交互用户数
    val user_rdd5 = user_rdd3
      .filter(r => r._1._1 == r._1._2) // ((item i, item i), count)
      .map(r => (r._1._1, r._2)) // (item i, N(i)) N(i)表示物品i被多少用户交互过
    user_rdd5.persist(StorageLevel.MEMORY_ONLY_SER)


    val user_rdd6 = user_rdd3
      .filter(r => r._1._1 != r._1._2) // ((item i,item j), count) C(i,j) 共现矩阵 i!=j
      .map(r => (r._1._1, (r._1._1, r._1._2, r._2))) // (item i, (item i, item j, count))
      .join(user_rdd5) // (item i, ((item i, item j, count), N(i)))
      .map(r => (r._2._1._2, (r._2._1._1, r._2._1._2, r._2._1._3, r._2._2))) // (item j, (item i, item j, count, N(i)))

    val user_rdd7 = user_rdd6
      .join(user_rdd5) // (item j, ((item i, item j, count, N(i)), N(j)))
      .map(r => (r._2._1._1, r._2._1._2, r._2._1._3, r._2._1._4, r._2._2)) // (item i,item j, count, N(i), N(j))

    // 计算同现相似度
    val similarity = user_rdd7
      .map(r => (r._1, r._2, r._3 / math.sqrt(r._4 * r._5))) // (item i,item j, count/sqrt(N(i) * N(j)) )

    similarity
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("ItemCF").master("spark://Future:7077").getOrCreate()

    val filepath = "/home/hadoop/Runs/data/small.dat"
    val data = load_data(spark, filepath, "::")

    val item_sims = item_similarity(data)
    println(item_sims.take(10))

  }

}
