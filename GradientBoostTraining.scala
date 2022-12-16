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
      

    val cvGBT = new CrossValidator()
      .setEstimator(gbt)
      .setEvaluator(evaluator.setMetricName("rmse"))
      .setEstimatorParamMaps(paramGridGBT)
      .setNumFolds(3)

    val cvGBTModel = cvGBT.fit(train_data)
    val predictions = cvGBTModel.transform(test_data)

    // Estos cambios son para poder ver las metricas de MAE y  R2  junto al RMSE 
    val rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    val mae = evaluator.setMetricName("mae").evaluate(predictions)
    val r2 = evaluator.setMetricName("r2").evaluate(predictions)
    println(s"\nRoot Mean Squared Error (RMSE) on test data = $rmse")
    println(s"Root Absolute Error (RAE) on test data = $mae")
    println(s"R2 on test data= $r2 \n")

  }

}
