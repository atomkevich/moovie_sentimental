import Utils.{loadStopWords, plainTextToLemmas}
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.mllib.classification.{NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by atomkevich on 5/11/17.
  */
object Word2VecMain {
  def main(args: Array[String]): Unit = {


    val sparkConf = new SparkConf().
      setAppName("Films Sentiment Analyzer")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val spark = SparkSession.builder()
      .config(sparkConf).getOrCreate()
    import spark.implicits._

    val stopWords = sc.broadcast(loadStopWords).value

    val negative_review: RDD[(String, Double)] = sc.textFile("rt-polarity_neg.txt").map(x => (x, 0.0))
    val positive_review = sc.textFile("rt-polarity_pos.txt").map(x => (x, 1.0))


    val allData = negative_review.union(positive_review)
    val docs = allData.map(doc => plainTextToLemmas(doc._1, stopWords)).toDS().cache()
    docs.printSchema()
    val word2vecModel = new Word2Vec()
      .setInputCol("value")
      .setOutputCol("features")
      .fit(docs)

    val res: DataFrame = word2vecModel.transform(docs)

    val featuresRDD = res.select("features").rdd.map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
    val featuresWithLabels = allData.zip(featuresRDD).map{case ((doc, sentiment), features) =>
      val featureTransform = org.apache.spark.mllib.linalg.Vectors.dense(features.toArray)
      LabeledPoint(label = sentiment, features = featureTransform)}



    println("String Learning and evaluating models")
    val Array(x_train, x_test) = featuresWithLabels.randomSplit(Array(0.8, 0.2))
    val model = SVMWithSGD.train(x_train, 100)
    //val model = NaiveBayes.train(x_train, 1.0)
    val result = model.predict(x_test.map(_.features))

    println(s"10 samples:")
    x_test.zip(result) take 10 foreach println


    val positive_true = sc.longAccumulator("positive_true")
    val positive_false = sc.longAccumulator("positive_false")
    val negative_false = sc.longAccumulator("negative_false")
    val negative_true =  sc.longAccumulator("negative_true")
    x_test.zip(result).foreach({case (label, res) => {
      (label.label, res) match {
        case (0.0, 0.0) =>
          negative_true.add(1)
        case (1.0, 1.0) =>
          positive_true.add(1)
        case (0.0, 1.0) =>
          negative_false.add(1)
        case (1.0, 0.0) =>
          positive_false.add(1)
        case _ => {
          println("ERROR!!!")
        }
      }
    }})

    println("POSITIVE TRUE:   " + positive_true)
    println("POSITIVE FALSE:   " + positive_false)
    println("NEGATIVE TRUE:   " + negative_true)
    println("NEGATIVE FALSE:   " + negative_false)


    val accuracy = x_test.zip(result)
      .filter{case (label, predict) => label.label == predict}.count.toFloat / x_test.count
    println(s"Model Accuracy: $accuracy")
  }

}
