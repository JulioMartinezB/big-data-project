package upm.bd

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.Imputer
import scala.math.Pi

object TestPreProcessing {

  def dataPreProcessing(data: DataFrame, airports: DataFrame): DataFrame = {

    var features = data.select("Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime",
      "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime", "DepDelay", "Origin", "Dest", "Distance",
      "TaxiOut", "ArrDelay")


    val cols_int = Array("Month", "DayOfWeek", "CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "ArrDelay",
      "DepTime", "CRSDepTime", "CRSArrTime")
    val cols_str = Array("UniqueCarrier", "Origin", "Dest")
    val cols_cat = Array("Month", "DayOfWeek", "avg(DepDelay)", "OriginLat", "DestLat", "OriginLong", "DestLong")
    val cols_cont = Array("CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "DepTimeMin",
      "CRSDepTimeMin", "CRSArrTimeMin")

    // Convertimos los tipos, así los NA que no se pueden castear pasan a ser nulos e imputamos los faltantes
    for (col <- cols_int) {
      features = features.withColumn(col, features(col).cast("int"))
    }

    for (col <- cols_str) {
      features = features.withColumn(col, features(col).cast("string"))
    }

    features = features.withColumn("DepTimeMin", (features("DepTime") / 100).cast("int"))
    features = features.withColumn("DepTimeMin", (features("DepTimeMin") * 60 + features("DepTime") % 100).cast("int"))
    features = features.drop("DepTime")

    features = features.withColumn("CRSDepTimeMin", (features("CRSDepTime") / 100).cast("int"))
    features = features.withColumn("CRSDepTimeMin", (features("CRSDepTimeMin") * 60 + features("CRSDepTime") % 100).cast("int"))
    features = features.drop("CRSDepTime")

    features = features.withColumn("CRSArrTimeMin", (features("CRSArrTime") / 100).cast("int"))
    features = features.withColumn("CRSArrTimeMin", (features("CRSArrTimeMin") * 60 + features("CRSArrTime") % 100).cast("int"))
    features = features.drop("CRSArrTime")

    val airlines = data.groupBy("UniqueCarrier").agg(mean("DepDelay"))
    features = features.join(airlines, Seq("UniqueCarrier"), "left").drop("UniqueCarrier")
    features = features.drop("FlightNum", "TailNum")

    val origins = airports.select("iata", "lat", "long")
      .withColumnRenamed("iata", "Origin")
      .withColumnRenamed("lat", "OriginLat")
      .withColumnRenamed("long", "OriginLong")

    val destinations = origins.withColumnRenamed("Origin", "Dest")
      .withColumnRenamed("OriginLat", "DestLat")
      .withColumnRenamed("OriginLong", "DestLong")

    features = features.join(origins, Seq("Origin"), "left").drop("Origin")
    features = features.join(destinations, Seq("Dest"), "left").drop("Dest")

    val cols_double = Seq("OriginLat", "OriginLong", "DestLat", "DestLong")

    for (col <- cols_double) {
      features = features.withColumn(col, features(col).cast("double"))
    }

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

    features = features.na.drop(Seq("ArrDelay"))

    features = features.withColumn("MonthSine", sin(features("Month") * Pi / 6))
    features = features.withColumn("MonthCos", cos(features("Month") * Pi / 6))

    features = features.drop("Month")
    features = features.drop("DayOfMonth")


    //    // Total de nulos por columna
    //    for (columna <- cols) {
    //      val nulos = features.filter(col(columna).isNull).count()
    //      print(columna + ": ")
    //      print(nulos + "\n")
    //    }
    //
    //    // Minimo máximo, media, etc
    //    for (columna <- cols) {
    //      features.describe(columna).show()
    //    }
    //
    //    // Ver los tipos de las variables
    //    features.printSchema()
    //
    //    // Mostrar mayores y menores valores y apariciones (se puede optimizar)
    //    for (columna <- cols) {
    //      features.groupBy(columna).count().orderBy(columna).show()
    //      features.groupBy(columna).count().orderBy(desc(columna)).show()
    //    }


    //    features.write.format("csv").option("header", "true").save(".\\data\\test.csv")
    features

  }


  }
