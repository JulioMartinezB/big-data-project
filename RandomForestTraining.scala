import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object RandomForestTraining {

  def modelTraining(features: DataFrame) {

    val Array(train, test) = features.randomSplit(Array(0.9, 0.1), seed = 12345)
    val cols = train.drop("ArrDelay").columns

    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")

    val train_data = assembler.transform(train)
    val test_data = assembler.transform(test)

    val rf = new RandomForestRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")

    val paramGridRF = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(2, 6))
      .addGrid(rf.numTrees, Array(5, 20))
      .addGrid(rf.maxBins, Array(20, 60))
      .build()

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val cvRF = new CrossValidator()
      .setEstimator(rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGridRF)
      .setNumFolds(3)

    val cvRFModel = cvRF.fit(train_data)
    val predictions = cvRFModel.transform(test_data)

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

  }

}
