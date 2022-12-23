package upm.bd

import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.Imputer
import scala.math.Pi

object PreProcessing {

  def dataPreProcessing(data: DataFrame, airports: DataFrame, train: Boolean): DataFrame = {

    var features = data.select("Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime",
      "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime", "DepDelay", "Origin", "Dest", "Distance",
      "TaxiOut", "ArrDelay")



    val cols_int = Array("Month", "DayofMonth", "DayOfWeek", "CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "ArrDelay",
      "DepTime", "CRSDepTime", "CRSArrTime")
    val cols_str = Array("UniqueCarrier", "Origin", "Dest")
    
    // Convertimos los tipos, así los NA que no se pueden castear pasan a ser nulos e imputamos los faltantes
    for (col <- cols_int) {
      features = features.withColumn(col, features(col).cast("int"))
    }

    // for (col <- cols_str) {
    //   features = features.withColumn(col, features(col).cast("string"))
    // }




    // convert time columns into minutes the 
    val cols_min= Array("DepTimeMin", "CRSDepTimeMin", "CRSArrTimeMin")
    val cols_raw= Array("DepTime", "CRSDepTime", "CRSArrTime")

    for ((col, colraw) <- cols_min.zip(cols_raw)){

        features = features.withColumn(col, (features(colraw) / 100).cast("int"))
        features = features.withColumn(col, (features(col) * 60 + features(colraw) % 100).cast("int"))
        features = features.drop(colraw)

    }

    // features = features.withColumn("DepTimeMin", (features("DepTime") / 100).cast("int"))
    // features = features.withColumn("DepTimeMin", (features("DepTimeMin") * 60 + features("DepTime") % 100).cast("int"))
    // features = features.drop("DepTime")

    // features = features.withColumn("CRSDepTimeMin", (features("CRSDepTime") / 100).cast("int"))
    // features = features.withColumn("CRSDepTimeMin", (features("CRSDepTimeMin") * 60 + features("CRSDepTime") % 100).cast("int"))
    // features = features.drop("CRSDepTime")

    // features = features.withColumn("CRSArrTimeMin", (features("CRSArrTime") / 100).cast("int"))
    // features = features.withColumn("CRSArrTimeMin", (features("CRSArrTimeMin") * 60 + features("CRSArrTime") % 100).cast("int"))
    // features = features.drop("CRSArrTime")


    // calculate de average delay per company

    val airlines = data.groupBy("UniqueCarrier").agg(mean("DepDelay"))
    features = features.join(airlines, Seq("UniqueCarrier"), "left").drop("UniqueCarrier")
    features = features.drop("FlightNum", "TailNum")



    // adds the latitude and longitude of the origin and destiny

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


     if (train) {

        //    // Columnas a borrar outliers
        //    cols = Array("CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "DepTimeMin", "CRSDepTimeMin",
        //      "CRSArrTimeMin", "avg(DepDelay)")
        //
        //    // Borramos outliers
        //    for (columna <- cols) {
        //      val quantiles = features.stat.approxQuantile(columna, Array(0.25, 0.75), 0.01)
        //      val IQR = quantiles(1) - quantiles(0)
        //      val lowerRange = quantiles(0) - 3 * IQR
        //      val upperRange = quantiles(1) + 3 * IQR
        //      print(lowerRange + " " + upperRange)
        //      features.filter(features(columna) < lowerRange || features(columna) > upperRange).show()
        //      features = features.filter(features(columna) > lowerRange && features(columna) < upperRange)
        //    }

        features = features.na.drop

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

    features = features.withColumn("Spring", isSpring(col("Month")))
    features = features.withColumn("Summer", isSummer(col("Month")))
    features = features.withColumn("Autumn", isAutumn(col("Month")))
    features = features.withColumn("Winter", isWinter(col("Month")))
    features = features.drop("Month")
    features.show()


    // Nulls treatments

    //    // Total de nulos por columna
    //    for (columna <- cols) {
    //      val nulos = features.filter(col(columna).isNull).count()
    //      print(columna + ": ")
    //      print(nulos + "\n")
    //    }


   
    
    features = features.na.drop(Seq("ArrDelay"))
    features

  }
  def analysis(data: DataFrame) {

    val cols = data.columns

    // Ver los tipos de las variables
    data.printSchema()

     // Minimo máximo, media, etc
    for (columna <- cols) {
        data.describe(columna).show()
    }

    // Mostrar mayores y menores valores y apariciones (se puede optimizar)
    for (columna <- cols) {
        data.groupBy(columna).count().orderBy(columna).show()
        data.groupBy(columna).count().orderBy(desc(columna)).show()
    }

  }
}
