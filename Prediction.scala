package upm.bd

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidatorModel

object Prediction {

  def predict(test: DataFrame, option: Int) {

    val model = selectModel(option)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")

    val predictions = model.transform(test)

    val rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    val mae = evaluator.setMetricName("mae").evaluate(predictions)
    val r2 = evaluator.setMetricName("r2").evaluate(predictions)
    println(s"\nRoot Mean Squared Error (RMSE) on test data = $rmse")
    println(s"Root Absolute Error (RAE) on test data = $mae")
    println(s"R2 on test data= $r2 \n")

  }

  def selectModel(option: Int): CrossValidatorModel = {
    if (option == 1) {
      CrossValidatorModel.load(".\\models\\lrModel")
    }
    else if (option == 2) {
      CrossValidatorModel.load(".\\models\\rfModel")
    }
    else {
      CrossValidatorModel.load(".\\models\\gbtModel")
    }
  }

}
