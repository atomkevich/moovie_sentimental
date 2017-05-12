
import Utils.{loadStopWords, plainTextToLemmas}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
  * Created by atomkevich on 5/11/17.
  */
object TfIdfMain {

  def main(args: Array[String]): Unit = {


    val sparkConf = new SparkConf().
      setAppName("Films Sentiment Analyzer")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val stopWords = sc.broadcast(loadStopWords).value

    val negative_review: RDD[(String, Double)] = sc.textFile("rt-polarity_neg.txt").map(x => (x, 0.0))
    val positive_review = sc.textFile("rt-polarity_pos.txt").map(x => (x, 1.0))


    val allData = negative_review.union(positive_review)

    val tf = new HashingTF(1000).transform(allData.map(doc => plainTextToLemmas(doc._1, stopWords)))
    val idf = new IDF().fit(tf)
    val tf_idf = idf.transform(tf)
    val dataWithFeature = allData.zip(tf_idf)


    val splits = dataWithFeature.randomSplit(Array(0.8, 0.2), seed = 11L)

    val training: RDD[LabeledPoint] = splits(0)
      .map({ case ((doc, label), features) => LabeledPoint(label = label, features = features)})

    val test: RDD[(String, Double, Vector)] = splits(1)
      .map({ case ((doc, label), features) => (doc, label, features)})



    val model = NaiveBayes.train(training, 1.0)

    /*val hashingTF = new HashingTF(1000)
    val paramGrid = new ParamGridBuilder()
      .build()

    val pipeline = new Pipeline()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)


    cv.fit(training)*/
    val positive_true = sc.longAccumulator("positive_true")
    val positive_false = sc.longAccumulator("positive_false")
    val negative_false = sc.longAccumulator("negative_false")
    val negative_true =  sc.longAccumulator("negative_true")
    test.foreach({case (doc, label, features) => {
      val res = model.predict(features)
      (label, res) match {
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
      println("DOCUMENT:  " + doc + "     CORRECT:   " + label + "    PREDICT:   " + res)
    }})

    println("POSITIVE TRUE:   " + positive_true)
    println("POSITIVE FALSE:   " + positive_false)
    println("NEGATIVE TRUE:   " + negative_true)
    println("NEGATIVE FALSE:   " + negative_false)


    val predictedValues = model.predict(test.map(_._3))
    val predictedRes = test.zip(predictedValues)
      .map{case ((doc, expected, features), predicted) => (expected, predicted)}

    val accuracy = 1.0 * predictedRes.filter{case (expected, predicted) => expected == predicted}.count() / predictedRes.count()
    println("ACCURACY:     " + accuracy)
  }



}
