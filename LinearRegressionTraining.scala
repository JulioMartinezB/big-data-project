package upm.bd

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object LinearRegressionTraining {

    def modelTraining(features: DataFrame) {

      val Array(train, test) = features.randomSplit(Array(0.9, 0.1), seed = 12345)
      val cols = train.drop("ArrDelay").columns

      val assembler = new VectorAssembler()
        .setInputCols(cols)
        .setOutputCol("features")

      val train_data = assembler.transform(train)
      val test_data = assembler.transform(test)

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
        .setMetricName("rmse")

      val cvLR = new CrossValidator()
        .setEstimator(lr)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGridLR)
        .setNumFolds(3)

      val lrModel = cvLR.fit(train_data)
      val predictions = lrModel.transform(test_data)
      val rmseLR = evaluator.evaluate(predictions)

      val rmse = evaluator.evaluate(predictions)
      println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    }
}
