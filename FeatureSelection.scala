package upm.bd

import org.apache.spark.ml.feature.{UnivariateFeatureSelector, UnivariateFeatureSelectorModel, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{monotonically_increasing_id}

object FeatureSelection {

  def featureSelection(data: DataFrame,  training: Boolean): DataFrame = {

    if (training) {

      var cols = data.drop("ArrDelay").columns

      var assembler = new VectorAssembler()
        .setInputCols(cols)
        .setOutputCol("features")

      var features_train = assembler.transform(data)
      features_train = features_train.select("features", "ArrDelay")

      val selector = new UnivariateFeatureSelector()
        .setFeatureType("continuous")
        .setLabelType("continuous")
        .setSelectionMode("fwe")
        .setSelectionThreshold(0.05)
        .setFeaturesCol("features")
        .setLabelCol("ArrDelay")
        .setOutputCol("selectedFeatures")

      val selectorModel = selector.fit(features_train.select("features", "ArrDelay"))
      selectorModel.write.overwrite().save(".\\models\\featureSelectionModel")
      val feat_selected = selectorModel.selectedFeatures

      var train = data.drop("ArrDelay")
      val colslen = train.columns.length

      for (i <- 0 to colslen - 1) {
        if (!feat_selected.contains(i)) {
          train = train.drop(data.columns(i))
        }
      }

      train = train.withColumn("id", monotonically_increasing_id)
      train = train.join(data.withColumn("id", monotonically_increasing_id).select("id", "ArrDelay"), Seq("id"), "left").drop("id")

      cols = train.drop("ArrDelay").columns

      assembler = new VectorAssembler()
        .setInputCols(cols)
        .setOutputCol("features")

      train = assembler.transform(train)
      train = train.select("ArrDelay", "features")

      train

    } else {
      val selectorModel = UnivariateFeatureSelectorModel.load(".\\models\\featureSelectionModel")
      val feat_selected = selectorModel.selectedFeatures

      var test = data.drop("ArrDelay")
      val colslen = test.columns.length

      for (i <- 0 to colslen - 1) {
        if (!feat_selected.contains(i)) {
          test = test.drop(data.columns(i))
        }
      }

      test = test.withColumn("id", monotonically_increasing_id)
      test = test.join(data.withColumn("id", monotonically_increasing_id).select("id", "ArrDelay"), Seq("id"), "left").drop("id")
      println("Selected Features and ArrDelay")
      test.show()

      val cols = test.drop("ArrDelay").columns

      val assembler = new VectorAssembler()
        .setInputCols(cols)
        .setOutputCol("features")

      test = assembler.transform(test)
      test = test.select("ArrDelay", "features")

      test
    }

  }

}
