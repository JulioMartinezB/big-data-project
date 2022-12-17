package upm.bd

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}

object LinearRegressionTraining {

  def modelTraining(train: DataFrame) {

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val paramGridLR = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept, Array(true, false))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 0.8))
      .build()

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")

    val cvLR = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(evaluator.setMetricName("rmse"))
      .setEstimatorParamMaps(paramGridLR)
      .setNumFolds(3)

    val lrModel = cvLR.fit(train)

    lrModel.write.overwrite().save(".\\models\\lrModel")


  }
}
