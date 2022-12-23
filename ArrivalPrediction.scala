package upm.bd

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._

object ArrivalPrediction {

  def main(args : Array[String]) {

    Logger.getLogger("org").setLevel(Level.WARN)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("Arrival Prediction")

    val sc = new SparkContext(conf)
    sc.setLogLevel("warn")

    val spark = SparkSession
      .builder()
      .appName("Arrival Prediction")
      .getOrCreate()

    // Checking if there are arguments specified
    if (args.length == 0) {
      println("No arguments passed to the application")
      -1
    }

    // Checking if there is a csv file specified
    if (args.length > 0) {
      // Checking if the specified file exists
      if (FileValidator.fileExists(args(0)) == false) {
        println("El fichero no existe")
      } else if (FileValidator.fileIsCSV(args(0)) == false) {
          // Checking if the specified file is a csv
          println("File is not .csv")
      } else {
          val flights = spark.read
            .format("csv")
            .option("delimiter", ",")
            .option("header", true)
            .load(args(0))
          // Checking if the file has the specified format of columns
          if (FileValidator.fileContainsColumns(flights) == false) {
            -1
          }
      }
    }

    // We read the specified file
    val flights = spark.read
      .format("csv")
      .option("delimiter", ",")
      .option("header", true)
      .load(args(0))

    // Auxiliar file with the locations of the airports
    val airports = spark.read
      .format("csv")
      .option("delimiter", ",")
      .option("header", true)
      .load("data/airports.csv")

    var training = false
    var model = 1
    var featSel = 1
    var analysis = false

    // We check the different arguments passed
    if (args.length > 1) {
      for (i <- 1 to args.length -1) {
        // In case of -y a already trained model will be used
        if (args(i) == "-y") {
          training = false
        }
        // In case of -n a new model will be trained
        else if (args(i) == "-n") {
          training = true
        }
        // In case of -lr Linear regression will be used
        else if (args(i) == "-lr") {
          model = 1
        }
        // In case of -rf Random Forest will be used
        else if (args(i) == "-rf") {
          model = 2
        }
        // In case of -gb Gradient Boosting will be used
        else if (args(i) == "-gb") {
          model = 3
        }
        // In case of -ufs Univariate Feature Selection will be used
        else if (args(i) == "-ufs") {
          featSel = 2
        }
        // In case of -pca Principal Component Analysis will be used
        else if (args(i) == "-pca") {
          featSel = 3
        } else if (args(i) == "-an") {
          analysis = true
        } else {
          println("Incorrect arguments")
          -1
        }
      }
    }

    // Case of training a new model
    if (training) {

      var Array(train, test) = flights.randomSplit(Array(0.8, 0.2), seed = 12345)
      train = PreProcessing.dataPreProcessing(train, airports, true, analysis)
      test = PreProcessing.dataPreProcessing(test, airports, false, false)

      // Without feature selection
      if (featSel == 1) {

        val cols = train.drop("ArrDelay").columns
        var assembler = new VectorAssembler()
          .setInputCols(cols)
          .setOutputCol("features")

        test = assembler.transform(test)
        train = assembler.transform(train)
        test = test.select("ArrDelay", "features")
        train = train.select("ArrDelay", "features")

        // Linear Regression
        if (model == 1) {
          LinearRegressionTraining.modelTraining(train)
          Prediction.predict(test, 1)
        }
        // Random Forest
        if (model == 2) {
          RandomForestTraining.modelTraining(train)
          Prediction.predict(test, 2)
        }
        // Gradient Boosting
        if (model == 3) {
          GradientBoostTraining.modelTraining(train)
          Prediction.predict(test, 3)
        }
      }

      // Using Univariate Feature Selection
      else if (featSel == 2) {

        val train_selected = FeatureSelection.featureSelection(train, true)
        val test_selected = FeatureSelection.featureSelection(test, false)

        // Linear Regression
        if (model == 1) {
          LinearRegressionTraining.modelTraining(train_selected)
          Prediction.predict(test_selected, 1)
        }
        // Random Forest
        if (model == 2) {
          RandomForestTraining.modelTraining(train_selected)
          Prediction.predict(test_selected, 2)
        }
        // Gradient Boosting
        if (model == 3) {
          GradientBoostTraining.modelTraining(train_selected)
          Prediction.predict(test_selected, 3)
        }
      }

      // Using PCA
      else if (featSel == 3) {
        val train_pca = PCASelection.nBestPCA(train, true)
        val test_pca = PCASelection.nBestPCA(test, false)

        // Linear Regression
        if (model == 1) {
          LinearRegressionTraining.modelTraining(train_pca)
          Prediction.predict(test_pca, 1)
        }
        // Random Forest
        if (model == 2) {
          RandomForestTraining.modelTraining(train_pca)
          Prediction.predict(test_pca, 2)
        }
        // Gradient Boosting
        if (model == 3) {
          GradientBoostTraining.modelTraining(train_pca)
          Prediction.predict(test_pca, 3)
        }

      }

    // Using an already trained model
    } else {

      var test = PreProcessing.dataPreProcessing(flights, airports, false, false)

      // Without feature selection
      if (featSel == 1) {

        val cols = test.drop("ArrDelay").columns
        var assembler = new VectorAssembler()
          .setInputCols(cols)
          .setOutputCol("features")

        test = assembler.transform(test)
        test = test.select("ArrDelay", "features")

        if (model == 1) {
          Prediction.predict(test, 1)
        }
        if (model == 2) {
          Prediction.predict(test, 2)
        }
        if (model == 3) {
          Prediction.predict(test, 3)
        }

      }

      // Using Univariate Feature Selection
      else if (featSel == 2) {

        val test_selected = FeatureSelection.featureSelection(test, false)

        // Linear Regression
        if (model == 1) {
          Prediction.predict(test_selected, 1)
        }
        // Random forest
        if (model == 2) {
          Prediction.predict(test_selected, 2)
        }
        // Gradient Boosting
        if (model == 3) {
          Prediction.predict(test_selected, 3)
        }

      }

      // Using PCA
      else if (featSel == 3)  {

        val test_pca = PCASelection.nBestPCA(test, false)

        // Linear Regression
        if (model == 1) {
          Prediction.predict(test_pca, 1)
        }
        // Random Forest
        if (model == 2) {
          Prediction.predict(test_pca, 2)
        }
        // Gradient Boosting
        if (model == 3) {
          Prediction.predict(test_pca, 3)
        }


      }

    }

    
  }
}