package upm.bd

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Row, SparkSession}

import scala.math.Pi
import org.apache.spark.sql.functions._
import org.sparkproject.dmg.pmml.False

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

    import spark.implicits._

    if (args.length == 0) {
      println("No arguments passed to the application")
      -1
    }

    if (args.length > 0) {
      if (FileValidator.fileExists(args(0)) == false) {
        println("El fichero no existe")
      } else if (FileValidator.fileIsCSV(args(0)) == false) {
          println("File is not .csv")
      } else {
          val flights = spark.read
            .format("csv")
            .option("delimiter", ",")
            .option("header", true)
            .load(args(0))
          if (FileValidator.fileContainsColumns(flights) == false) {
            -1
          }
      }
    }

    val flights = spark.read
      .format("csv")
      .option("delimiter", ",")
      .option("header", true)
      .load(args(0))

    var airports = spark.read
      .format("csv")
      .option("delimiter", ",")
      .option("header", true)
      .load("C:\\Users\\carvs\\Documentos\\Master\\Primer cuatri\\Big data\\practical-work\\data\\airports.csv")

    var training = false
    var model = 1

    if (args.length == 2) {
      if (args(1) == "-y") {
        training = false
      }
      else if (args(1) == "-n") {
        training = true
      }
      else if (args(1) == "-lr"){
        model = 1
      } else if (args(1) == "-rf") {
        model = 2
      } else if (args(1) == "-gb") {
        model = 3
      } else {
        println("If you specify 2 args, the second must be -y to use a pretrained model, -n to train a new model, -lr for linear" +
          "regression, -rf for Random Forest or -gb for Gradient Boosting")
        -1
      }
    }

    if (args.length > 2) {
      if (args(1) == "-y") {
        training = false
      }
      else if (args(1) == "-n") {
        training = true
      }
      else {
        println("If you specify  3 args, the second arg it must be -y to use a pretrained model or -n to train a new model")
        -1
      }
      if (args(2) == "-lr") {
        model = 1
      } else if (args(2) == "-rf") {
        model = 2
      } else if (args(2) == "-gb") {
        model = 3
      } else {
        println("If you specify 3 args, the third must be -y to use a pretrained model, -n to train a new model, -lr for linear" +
          "regression, -rf for Random Forest or -gb for Gradient Boosting")
        -1
      }
    }

    if (training) {
      var Array(train, test) = flights.randomSplit(Array(0.8, 0.2), seed = 12345)
      train = TrainingPreProcessing.dataPreProcessing(train, airports)
      test = TestPreProcessing.dataPreProcessing(test, airports)
      FeatureSelection.featureSelection(train)
      val train_pca = PCASelection.nBestPCA(train, true)
      val test_pca = PCASelection.nBestPCA(test, false)
      if (model == 1) {
        LinearRegressionTraining.modelTraining(train_pca)
        Prediction.predict(test_pca, 1)
      }
      if (model == 2) {
        RandomForestTraining.modelTraining(train_pca)
        Prediction.predict(test_pca, 2)
      }
      if (model == 3) {
        GradientBoostTraining.modelTraining(train_pca)
        Prediction.predict(test_pca, 3)
      }
    } else {
      val test = TestPreProcessing.dataPreProcessing(flights, airports)
      val test_pca = PCASelection.nBestPCA(test, false)
      if (model == 1) {
        Prediction.predict(test_pca, 1)
      }
      if (model == 2) {
        Prediction.predict(test_pca, 2)
      }
      if (model == 3) {
        Prediction.predict(test_pca, 3)
      }
    }


//    var flights = spark.read
//                      .format("csv")
//                      .option("delimiter", ",")
//                      .option("header", true)
//                      .load("C:\\Users\\carvs\\Documentos\\Master\\Primer cuatri\\Big data\\practical-work\\data\\features.csv")
//
//    flights.show()
//
//    val cols_int = Seq("DayOfWeek", "CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "DepTimeMin", "CRSDepTimeMin",
//                      "CRSArrTimeMin", "avg(DepDelay)","ArrDelay")
//    val cols_double = Seq("OriginLat", "OriginLong", "DestLat", "DestLong", "MonthSine", "MonthCos")
//
//    for (col <- cols_int) {
//      flights = flights.withColumn(col, flights(col).cast("int"))
//    }
//
//    for (col <- cols_double) {
//      flights = flights.withColumn(col, flights(col).cast("double"))
//    }
//
//    FeatureSelection.featureSelection(flights)






    //LinearRegressionTraining.modelTraining(flights)


//    val flights = spark.read
//                  .format("csv")
//                  .option("delimiter", ",")
//                  .option("header", true)
//                  .load("C:\\Users\\carvs\\Documentos\\Master\\Primer cuatri\\Big data\\practical-work\\data\\2005.csv")
//
//    var airports = spark.read
//      .format("csv")
//      .option("delimiter", ",")
//      .option("header", true)
//      .load("C:\\Users\\carvs\\Documentos\\Master\\Primer cuatri\\Big data\\practical-work\\data\\airports.csv")
//
//    val airlines = flights.groupBy("UniqueCarrier")
//                           .agg(mean("DepDelay"))
//
//    var features = flights.select("Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime",
//                              "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime", "DepDelay",
//                              "Origin", "Dest", "Distance", "TaxiOut", "Cancelled", "CancellationCode", "ArrDelay")
//
//
//    features = features.withColumn("DepTimeMin", (features("DepTime") / 100).cast("int"))
//    features = features.withColumn("DepTimeMin", (features("DepTimeMin") * 60 + features("DepTime") % 100).cast("int"))
//    features = features.drop("DepTime")
//
//    features = features.withColumn("CRSDepTimeMin", (features("CRSDepTime") / 100).cast("int"))
//    features = features.withColumn("CRSDepTimeMin", (features("CRSDepTimeMin") * 60 + features("CRSDepTime") % 100).cast("int"))
//    features = features.drop("CRSDepTime")
//
//    features = features.withColumn("CRSArrTimeMin", (features("CRSArrTime") / 100).cast("int"))
//    features = features.withColumn("CRSArrTimeMin", (features("CRSArrTimeMin") * 60 + features("CRSArrTime") % 100).cast("int"))
//    features = features.drop("CRSArrTime")
//
//
//    features = features.join(airlines, Seq("UniqueCarrier"), "left").drop("UniqueCarrier")
//
//    features = features.drop("FlightNum", "TailNum")
//
//    airports = airports.select("iata", "lat", "long")
//                        .withColumnRenamed("iata", "Origin")
//                        .withColumnRenamed("lat", "OriginLat")
//                        .withColumnRenamed("long", "OriginLong")
//
//    features = features.join(airports, Seq("Origin"), "left").drop("Origin")
//
//    airports = airports.withColumnRenamed("Origin", "Dest")
//      .withColumnRenamed("OriginLat", "DestLat")
//      .withColumnRenamed("OriginLong", "DestLong")
//
//    features = features.join(airports, Seq("Dest"), "left").drop("Dest")
//    //features.join(airlines, features.Origin=airlines.iata)
//
//    features = features.na.drop(Seq("ArrDelay")).drop("Cancelled").drop("CancellationCode")
//
//    features = features.withColumn("MonthSine", sin(features("Month")*Pi/6))
//    features = features.withColumn("MonthCos", cos(features("Month")*Pi/6))
//
//    features = features.drop("Month")
//    features = features.drop("DayOfMonth")
//
//    val cols_int = Seq("DayOfWeek", "CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "ArrDelay")
//    val cols_double = Seq("OriginLat", "OriginLong", "DestLat", "DestLong")
//
//    for (col <- cols_int) {
//      features = features.withColumn(col, features(col).cast("int"))
//    }
//
//    for (col <- cols_double) {
//      features = features.withColumn(col, features(col).cast("double"))
//    }
//
//    var cols  = features.columns
//
////    for (columna <- cols) {
////      features.describe(columna).show()
////    }
//
//
//
//    //features.printSchema()
//
////    for (columna <- cols) {
////      var ordenado = features.groupBy(columna).count().orderBy(columna).show()
////      features.groupBy(columna).count().orderBy(desc(columna)).show()
////    }
////
//
//    features = features.na.drop
//
//
////    for (columna <- cols) {
////      val nulos = features.filter(col(columna).isNull).count()
////      print(columna + ": ")
////      print(nulos + "\n")
////    }
//
//    cols = Array("CRSElapsedTime", "DepDelay", "Distance","TaxiOut","ArrDelay","DepTimeMin","CRSDepTimeMin",
//                  "CRSArrTimeMin",  "avg(DepDelay)")
//    for (columna <- cols) {
//      val quantiles = features.stat.approxQuantile(columna, Array(0.25, 0.75), 0.01)
//      val IQR = quantiles(1) - quantiles(0)
//      val lowerRange = quantiles(0) - 3 * IQR
//      val upperRange = quantiles(1) + 3 * IQR
//      print(lowerRange + " " + upperRange)
//      features.filter(features(columna) < lowerRange || features(columna) > upperRange).show()
//      features = features.filter(features(columna) > lowerRange && features(columna) < upperRange)
//    }
//
//    print("RWOWOREJOWRPEFPWJFPWEJFPWEOJFPIWIE   ")
//    print(features.count())
//
//    features.write.format("csv").option("header", "true").save(".\\data\\features2.csv")
//
//
//



    
  }
}