import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import scala.math.Pi

object DataPreProcessing {

  def dataPreProcessing(data: DataFrame, airports: DataFrame) {

    var features = data.select("Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime",
      "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime", "DepDelay", "Origin", "Dest", "Distance",
      "TaxiOut", "Cancelled", "CancellationCode", "ArrDelay")

    val airlines = data.groupBy("UniqueCarrier").agg(mean("DepDelay"))
    features = features.join(airlines, Seq("UniqueCarrier"), "left").drop("UniqueCarrier")
    features = features.drop("FlightNum", "TailNum")

    features = features.withColumn("DepTimeMin", (features("DepTime") / 100).cast("int"))
    features = features.withColumn("DepTimeMin", (features("DepTimeMin") * 60 + features("DepTime") % 100).cast("int"))
    features = features.drop("DepTime")

    features = features.withColumn("CRSDepTimeMin", (features("CRSDepTime") / 100).cast("int"))
    features = features.withColumn("CRSDepTimeMin", (features("CRSDepTimeMin") * 60 + features("CRSDepTime") % 100).cast("int"))
    features = features.drop("CRSDepTime")

    features = features.withColumn("CRSArrTimeMin", (features("CRSArrTime") / 100).cast("int"))
    features = features.withColumn("CRSArrTimeMin", (features("CRSArrTimeMin") * 60 + features("CRSArrTime") % 100).cast("int"))
    features = features.drop("CRSArrTime")

    val origins = airports.select("iata", "lat", "long")
                        .withColumnRenamed("iata", "Origin")
                        .withColumnRenamed("lat", "OriginLat")
                        .withColumnRenamed("long", "OriginLong")

    val destinations = origins.withColumnRenamed("Origin", "Dest")
      .withColumnRenamed("OriginLat", "DestLat")
      .withColumnRenamed("OriginLong", "DestLong")

    features = features.join(origins, Seq("Origin"), "left").drop("Origin")
    features = features.join(destinations, Seq("Dest"), "left").drop("Dest")

    features = features.na.drop(Seq("ArrDelay")).drop("Cancelled").drop("CancellationCode")

    features = features.withColumn("MonthSine", sin(features("Month")*Pi/6))
    features = features.withColumn("MonthCos", cos(features("Month")*Pi/6))

    features = features.drop("Month")
    features = features.drop("DayOfMonth")

    var cols  = features.columns
    val cols_int = Seq("DayOfWeek", "CRSElapsedTime", "DepDelay", "Distance", "TaxiOut", "ArrDelay")
    val cols_double = Seq("OriginLat", "OriginLong", "DestLat", "DestLong")

    // Convertimos los tipos, así los NA que no se pueden castear pasan a ser nulos
    for (col <- cols_int) {
      features = features.withColumn(col, features(col).cast("int"))
    }

    for (col <- cols_double) {
      features = features.withColumn(col, features(col).cast("double"))
    }

    features = features.na.drop

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


    // Columnas a borrar outliers
    cols = Array("CRSElapsedTime", "DepDelay", "Distance","TaxiOut","ArrDelay","DepTimeMin","CRSDepTimeMin",
                  "CRSArrTimeMin",  "avg(DepDelay)")

    // Borramos outliers
    for (columna <- cols) {
      val quantiles = features.stat.approxQuantile(columna, Array(0.25, 0.75), 0.01)
      val IQR = quantiles(1) - quantiles(0)
      val lowerRange = quantiles(0) - 3 * IQR
      val upperRange = quantiles(1) + 3 * IQR
      print(lowerRange + " " + upperRange)
      features.filter(features(columna) < lowerRange || features(columna) > upperRange).show()
      features = features.filter(features(columna) > lowerRange && features(columna) < upperRange)
    }

    features.write.format("csv").option("header", "true").save(".\\data\\features2.csv")


  }

}
