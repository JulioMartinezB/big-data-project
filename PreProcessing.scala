package upm.bd

import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.Imputer
import scala.math.Pi

object PreProcessing {

  def dataPreProcessing(data: DataFrame, airports: DataFrame, train: Boolean, do_analysis: Boolean): DataFrame = {

    var features = data.select("Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime",
      "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime", "DepDelay", "Origin", "Dest", "Distance",
      "TaxiOut", "ArrDelay")

    // Numeric columns to cast so the NA values that are string are converted to nulls
    val cols_int = Array("Month", "DayofMonth", "DayOfWeek", "CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "ArrDelay",
      "DepTime", "CRSDepTime", "CRSArrTime")
    
    // Cast of numeric columns
    for (col <- cols_int) {
      features = features.withColumn(col, features(col).cast("int"))
    }

    // Convert time columns into minutes the
    val cols_min= Array("DepTimeMin", "CRSDepTimeMin", "CRSArrTimeMin")
    val cols_raw= Array("DepTime", "CRSDepTime", "CRSArrTime")

    for ((col, colraw) <- cols_min.zip(cols_raw)){
        features = features.withColumn(col, (features(colraw) / 100).cast("int"))
        features = features.withColumn(col, (features(col) * 60 + features(colraw) % 100).cast("int"))
        features = features.drop(colraw)
    }


    // Calculate de average delay per company

    val airlines = data.groupBy("UniqueCarrier").agg(mean("DepDelay"))
    features = features.join(airlines, Seq("UniqueCarrier"), "left").drop("UniqueCarrier")
    features = features.drop("FlightNum", "TailNum")



    // Adds the latitude and longitude of the origin and destiny

    val origins = airports.select("iata", "lat", "long")
      .withColumnRenamed("iata", "Origin")
      .withColumnRenamed("lat", "OriginLat")
      .withColumnRenamed("long", "OriginLong")

    val destinations = origins.withColumnRenamed("Origin", "Dest")
      .withColumnRenamed("OriginLat", "DestLat")
      .withColumnRenamed("OriginLong", "DestLong")

    features = features.join(origins, Seq("Origin"), "left").drop("Origin")
    features = features.join(destinations, Seq("Dest"), "left").drop("Dest")

    // We cast now the new features that are of type double
    val cols_double = Seq("OriginLat", "OriginLong", "DestLat", "DestLong")

    for (col <- cols_double) {
      features = features.withColumn(col, features(col).cast("double"))
    }

    // Previous data analysis
    if (do_analysis) {
      analysis(features)
    }

     // In case of training data we will drop the outliers and the rows with missing values
     if (train) {

        // Columns in which we will drop outliers for the training proecess
        val cols = Array("CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "DepTimeMin", "CRSDepTimeMin",
          "CRSArrTimeMin", "avg(DepDelay)")

        // Dropping outliers
        for (columna <- cols) {
          val quantiles = features.stat.approxQuantile(columna, Array(0.25, 0.75), 0.01)
          val IQR = quantiles(1) - quantiles(0)
          val lowerRange = quantiles(0) - 3 * IQR
          val upperRange = quantiles(1) + 3 * IQR
          features = features.filter(features(columna) > lowerRange && features(columna) < upperRange)
        }

        // we drop the missing values
        features = features.na.drop

     // In case of test data we will impute the missing values with mean for continuous and mode for cathegorical
     } else {

        val cols_cat = Array("DayOfWeek", "avg(DepDelay)", "OriginLat", "DestLat", "OriginLong", "DestLong")
        val cols_cont = Array("CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "DepTimeMin",
        "CRSDepTimeMin", "CRSArrTimeMin")

        val imputer_cont = new Imputer()
        .setInputCols(cols_cont)
        .setOutputCols(cols_cont)
        .setStrategy("mean")

        val imputer_cat = new Imputer()
        .setInputCols(cols_cat)
        .setOutputCols(cols_cat)
        .setStrategy("mode")

        val model_cat = imputer_cat.fit(features)
        val model_cont = imputer_cont.fit(features)
        features = model_cat.transform(features)
        features = model_cont.transform(features)

    }

    
    // UDFS to create binary columns for each season
    val isSpring = udf((col: Int) => {
      if (col == 3 | col == 4 | col == 5) 1
      else 0
    })

    val isSummer = udf((col: Int) => {
      if (col == 6 | col == 7 | col == 8) 1
      else 0
    })

    val isAutumn = udf((col: Int) => {
      if (col == 9 | col == 10 | col == 11) 1
      else 0
    })

    val isWinter = udf((col: Int) => {
      if (col == 12 | col == 1 | col == 2) 1
      else 0
    })

    // New binary columns indicating the season
    features = features.withColumn("Spring", isSpring(col("Month")))
    features = features.withColumn("Summer", isSummer(col("Month")))
    features = features.withColumn("Autumn", isAutumn(col("Month")))
    features = features.withColumn("Winter", isWinter(col("Month")))
    features = features.drop("Month")
   
    // If there is misssing value for ArrDelay we drop the row because we have no information
    // to check the correctness of the model
    features = features.na.drop(Seq("ArrDelay"))
    features

  }

  // Analysys of the dataset (higher and lower values and types for columns)
  def analysis(data: DataFrame) {

    val cols = data.columns

    // type of the columns
    println("Types of the columns")
    data.printSchema()

     // Min, max and mean
    for (columna <- cols) {
      println("Description of column " + columna)
      data.describe(columna).show()
    }

    // Highest and lowest values (to check possible outliers)
    for (columna <- cols) {
      println("Lowest values of column " + columna)
      data.groupBy(columna).count().orderBy(columna).show()
      println("Highest values of column " + columna)
      data.groupBy(columna).count().orderBy(desc(columna)).show()
    }

  }
}
