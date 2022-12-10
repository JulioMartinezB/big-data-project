package upm.bd

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{SparkSession, Row}
import scala.math.Pi
import org.apache.spark.sql.functions._

object ArrivalPrediction {

  def main(args : Array[String]) {

    Logger.getLogger("org").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName("My first Spark application")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    sc.setLogLevel("warn")

    val flights = spark.read
                  .format("csv")
                  .option("delimiter", ",")
                  .option("header", true)
                  .load("C:\\Users\\carvs\\OneDrive - Universidad Politécnica de Madrid\\Documentos\\Master\\Primer cuatri\\Big data\\practical-work\\data\\2008.csv")

    var airports = spark.read
      .format("csv")
      .option("delimiter", ",")
      .option("header", true)
      .load("C:\\Users\\carvs\\OneDrive - Universidad Politécnica de Madrid\\Documentos\\Master\\Primer cuatri\\Big data\\practical-work\\data\\airports.csv")

    val airlines = flights.groupBy("UniqueCarrier")
                           .agg(mean("DepDelay"))

    var features = flights.select("Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime",
                              "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime", "DepDelay",
                              "Origin", "Dest", "Distance", "TaxiOut", "Cancelled", "CancellationCode", "ArrDelay")

    features = features.withColumn("DepTimeMin", (features("DepTime") / 100).cast("int"))
    features = features.withColumn("DepTimeMin", (features("DepTimeMin") * 60 + features("DepTime") % 100).cast("int"))
    features = features.drop("DepTime")

    features = features.withColumn("CRSDepTimeMin", (features("CRSDepTime") / 100).cast("int"))
    features = features.withColumn("CRSDepTimeMin", (features("CRSDepTimeMin") * 60 + features("CRSDepTime") % 100).cast("int"))
    features = features.drop("CRSDepTime")

    features = features.withColumn("CRSArrTimeMin", (features("CRSArrTime") / 100).cast("int"))
    features = features.withColumn("CRSArrTimeMin", (features("CRSArrTimeMin") * 60 + features("CRSArrTime") % 100).cast("int"))
    features = features.drop("CRSArrTime")

    features = features.join(airlines, Seq("UniqueCarrier"), "left").drop("UniqueCarrier")

    features = features.drop("FlightNum", "TailNum")

    airports = airports.select("iata", "lat", "long")
                        .withColumnRenamed("iata", "Origin")
                        .withColumnRenamed("lat", "OriginLat")
                        .withColumnRenamed("long", "OriginLong")

    features = features.join(airports, Seq("Origin"), "left").drop("Origin")

    airports = airports.withColumnRenamed("Origin", "Dest")
      .withColumnRenamed("OriginLat", "DestLat")
      .withColumnRenamed("OriginLong", "DestLong")

    features = features.join(airports, Seq("Dest"), "left").drop("Dest")
    //features.join(airlines, features.Origin=airlines.iata)

    features = features.na.drop(Seq("ArrDelay")).drop("Cancelled").drop("CancellationCode")

    features = features.withColumn("MonthSine", sin(features("Month")*Pi/6))
    features = features.withColumn("MonthCos", cos(features("Month")*Pi/6))

    features = features.drop("Month")
    features = features.drop("DayOfMonth")

    features.show()

    airlines.show()

    val months = Seq((1,Pi/6), (2,2*Pi/6), (3,3*Pi/6), (4,4*Pi/6), (5,5*Pi/6), (6,6*Pi/6),
                 (7,7*Pi/6), (8,8*Pi/6), (9,9*Pi/6), (10,10*Pi/6), (11,11*Pi/6), (12,12*Pi/6))



//    val pagesTuples = pageCounts.flatMap(line => {
//      val tokens = line.split(" ")
//      if (tokens.size == 4) {
//        List((tokens(0), tokens(1), tokens(2).toLong, tokens(3).toLong))
//      } else {
//        List()
//      }
//    })
//
//    val mostVisited = pagesTuples.reduce((t1,t2) => if (t1._3 > t2._3) t1._3 else t2._3)
//    val leastVisited = pagesTuples.reduce((t1,t2) => if (t1._3 < t2._3) t1._3 else t2._3)
//
//    val dist_bin = (mostVisited - leastVisited) / 20
//
//    val bin_values = pagesTuples.flatMap((tuple => {
//      List(tuple._3 % dist_bin, 1)
//    }))

//    // Another option. Lower complexity
//    val maxReq = enPagesTuples.map(_._3)
//      .reduce(math.max(_, _))
//    val mostReq = enPagesTuples.filter(_._3 == maxReq).first
//    println(mostReq._2 + "\t" + mostReq._3)
//
//    // Yet another, even better option
//    val mostReq = enPagesTuples.reduce((t1, t2) => if (t1._3 > t2._3) t1 else t2)
//    println(mostReq._2 + "\t" + mostReq._3)


    
  }
}