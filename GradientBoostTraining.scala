import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object GradientBoostTraining {

  def modelTraining(features: DataFrame) {

    val Array(train, test) = features.randomSplit(Array(0.9, 0.1), seed = 12345)
    val cols = train.drop("ArrDelay").columns

    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")

    val train_data = assembler.transform(train)
    val test_data = assembler.transform(test)

    val gbt = new GBTRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")
      .setMaxIter(10)

    val paramGridGBT = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, Array(2, 6))
      .addGrid(gbt.maxBins, Array(20, 60))
      .build()

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val cvGBT = new CrossValidator()
      .setEstimator(gbt)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGridGBT)
      .setNumFolds(3)

    val cvGBTModel = cvGBT.fit(train_data)
    val predictions = cvGBTModel.transform(test_data)

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

  }

}
