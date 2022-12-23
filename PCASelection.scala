package upm.bd

import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{PCA, PCAModel, StandardScaler, VectorAssembler, VectorSlicer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{Row, SparkSession}

object PCASelection {

  def nBestPCA(data: DataFrame, train: Boolean): DataFrame = {


    val cols = data.drop("ArrDelay").columns

    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")

    var data_assembled = assembler.transform(data)
    data_assembled = data_assembled.select("ArrDelay", "features")

    // Scalling features before PCA
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val data_scaled = scaler.fit(data_assembled)
      .transform(data_assembled)
      .select("scaledFeatures", "ArrDelay")

    //data_scaled.show(false)

    // In case of training we generate a PCA model and save it for posterior use
    if (train) {

      val pca = new PCA()
        .setInputCol("scaledFeatures")
        .setOutputCol("pcaFeatures")
        .setK(cols.size)

      val pcaModel = pca.fit(data_scaled)
      pcaModel.write.overwrite().save(".\\models\\pcaModel")
      get_result(pcaModel, data_scaled, false)

    // In case of just predicting we use an already PCA model
    } else {
      val pcaModel = PCAModel.load(".\\models\\pcaModel")
      get_result(pcaModel, data_scaled, true)

    }

  }

  // we get the principal components and select the number that explain a certain amount of variance
  def get_result(model: PCAModel, data_scaled: DataFrame, print: Boolean): DataFrame = {

    val result = model.transform(data_scaled).select("pcaFeatures", "ArrDelay")
    result.show(false)

    val spark = SparkSession
      .builder()
      .appName("Arrival Prediction")
      .getOrCreate()

    import spark.implicits._

    // We check for the number of components that explain the 95% of the variance of the data
    val explained_var = model.explainedVariance
    val explained_var_df = explained_var.values.toSeq.toDF("explained_var")
    var cumsum_var = explained_var_df.withColumn("Cumulative Sum EV", sum("explained_var")
      .over(Window.orderBy($"explained_var".desc)))
    cumsum_var = cumsum_var.where(col("Cumulative Sum EV") <= 0.95)
    if (print) {
      cumsum_var.show(false)
      println("n selected", cumsum_var.count(), "\n")
    }


    val slicer = new VectorSlicer()
      .setInputCol("pcaFeatures")
      .setOutputCol("features")
      .setIndices(Array.range(0, cumsum_var.count().toInt))

    val pca_sliced = slicer.transform(result).select("features", "ArrDelay")
    pca_sliced

  }

}