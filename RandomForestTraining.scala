package upm.bd

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object RandomForestTraining {

  def modelTraining(train: DataFrame) {

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

    val cvRF = new CrossValidator()
      .setEstimator(rf)
      .setEvaluator(evaluator.setMetricName("rmse"))
      .setEstimatorParamMaps(paramGridRF)
      .setNumFolds(3)

    val rfModel = cvRF.fit(train)

    rfModel.write.overwrite().save(".\\models\\rfModel")


  }

}
