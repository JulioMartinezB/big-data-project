package upm.bd

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object GradientBoostTraining {

  def modelTraining(train: DataFrame) {

    val gbt = new GBTRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")

    val paramGridGBT = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, Array(2, 6))
      .addGrid(gbt.maxBins, Array(20, 60))
      .build()

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")

    val cvGBT = new CrossValidator()
      .setEstimator(gbt)
      .setEvaluator(evaluator.setMetricName("rmse"))
      .setEstimatorParamMaps(paramGridGBT)
      .setNumFolds(3)

    val gbtModel = cvGBT.fit(train)

    gbtModel.write.overwrite().save(".\\models\\gbtModel")

  }

}
