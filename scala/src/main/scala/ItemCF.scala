import com.naihai.study.Utils

import scala.collection.mutable


/**
  *
  * 基于物品的协同过滤算法
  */

object ItemCF {


  /**
    * 计算物品之间的相似度
    */
  def item_similarity(train_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]]):
  mutable.HashMap[Int, mutable.HashMap[Int, Double]] = {

    val start = System.currentTimeMillis()

    val item_sims: mutable.HashMap[Int, mutable.HashMap[Int, Double]] = mutable.HashMap()
    val co_items: mutable.HashMap[Int, mutable.HashMap[Int, Int]] = mutable.HashMap()
    val num_items: mutable.HashMap[Int, Int] = mutable.HashMap()

    // 统计共现矩阵
    train_data.foreach(elem => {
      elem._2.foreach(item_i => {
        if (!co_items.contains(item_i._1))
          co_items.put(item_i._1, mutable.HashMap[Int, Int]())

        if (!num_items.contains(item_i._1))
          num_items.put(item_i._1, 0)
        num_items(item_i._1) += 1

        elem._2.foreach(item_j => {
          if (!item_i._1.equals(item_j._1)) {
            if (!co_items(item_i._1).contains(item_j._1))
              co_items(item_i._1)(item_j._1) = 0
            co_items(item_i._1)(item_j._1) += 1
          }
        })
      })
    })

    //计算物品之间的相似度
    co_items.foreach(elem => {
      if (!item_sims.contains(elem._1))
        item_sims.put(elem._1, mutable.HashMap[Int, Double]())

      elem._2.foreach(item => {
        item_sims(elem._1).put(item._1, item._2 / math.sqrt(num_items(elem._1) * num_items(item._1)))
      })
    })
    println("calculate item similarity done, cost " + (System.currentTimeMillis() - start) + " ms")
    item_sims

  }

  /**
    * 执行推荐 为每个用户都推荐物品
    */
  def recommend(train_data: mutable.HashMap[Int, mutable.HashMap[Int, Int]],
                item_sims: mutable.HashMap[Int, mutable.HashMap[Int, Double]],
                nearest_k: Int, top_n: Int): mutable.HashMap[Int, List[Int]] = {

    val start = System.currentTimeMillis()
    val recommend_lists: mutable.HashMap[Int, List[Int]] = mutable.HashMap()

    train_data.foreach(elem => {
      val rank: mutable.HashMap[Int, Double] = mutable.HashMap()
      // 该用户交互过的物品elem._2
      elem._2.foreach(item => {
        //根据相似度矩阵, 找到距离物品i最近的K个物品
        val nearest_items = item_sims(item._1).toList.sortBy(_._2).reverse.take(nearest_k) //根据相似度排序
        nearest_items.foreach(e => {
          //该物品没有被用户交互过
          if (!elem._2.contains(e._1)) {
            if (!rank.contains(e._1))
              rank.put(e._1, 0)
            rank(e._1) += e._2 * item._2 //e._2是物品ij的相似度 item._2是用户对物品i的评分
          }
        })
      })

      // 选择Top N 作为该用户的推荐
      val top_n_items = rank.toList.sortBy(_._2).reverse.take(top_n).map(_._1)
      recommend_lists.put(elem._1, top_n_items)

    })
    println("recommend done, cost " + (System.currentTimeMillis() - start) + " ms")
    recommend_lists
  }

  def main(args: Array[String]): Unit = {

    val start = System.currentTimeMillis()

    val all_data = Utils.load_data("E:\\Python\\Projects\\Recommendation\\data\\ratings.dat")
    val (train, test) = Utils.split_data(all_data, 10, 1)

    // 计算物品相似度
    val W = item_similarity(train)

    // 获取推荐列表
    val recommends = recommend(train, W, nearest_k = 10, top_n = 10)

    // 计算指标
    val p = Utils.precision(train, test, recommends)

    val r = Utils.recall(train, test, recommends)

    val c = Utils.coverage(train, recommends)

    val po = Utils.popularity(train, recommends)

    println(p, r, c, po)

    println("all work  done, cost " + (System.currentTimeMillis() - start)/1000 + " s")

  }

}