package upm.bd

import org.apache.spark.ml.feature.{UnivariateFeatureSelector, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{monotonicallyIncreasingId, monotonically_increasing_id}

object FeatureSelection {

  def featureSelection(data: DataFrame) {

    var cols = data.drop("ArrDelay").columns

    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")

    var features = assembler.transform(data)

    features = features.select("features", "ArrDelay")

    val selector = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous")
      .setSelectionMode("numTopFeatures")
      .setSelectionThreshold(4)
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val result = selector.fit(features.select("features", "ArrDelay")).transform(features)

    println(s"UnivariateFeatureSelector output with top ${selector.getSelectionThreshold}" +
      s" features selected using f_classif")
    result.select("selectedFeatures").show(false)

  }

}
