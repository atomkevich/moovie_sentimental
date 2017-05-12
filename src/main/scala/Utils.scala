import java.io.InputStream
import java.util.Properties

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

/**
  * Created by atomkevich on 5/11/17.
  */
object Utils {
  def loadStopWords = {
    val stream: InputStream = getClass.getResourceAsStream("/stopwords.txt")
    val lines = scala.io.Source.fromInputStream(stream).getLines
    lines.toSet
  }

  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)
  }

  def plainTextToLemmas(text: String, stopWords: Set[String]): Seq[String] = {

    val pipeline = createNLPPipeline()
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala;
         token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && !stopWords.contains(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }
}
