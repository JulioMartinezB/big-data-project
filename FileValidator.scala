package upm.bd

import org.apache.spark.sql.{DataFrame, SparkSession}

import java.nio.file.{Files, Paths}

object FileValidator {

  def fileExists(path: String): Boolean = {

    if (Files.exists(Paths.get(path))) {
      true
    } else {
      false
    }

  }

  def fileIsCSV(path: String): Boolean = {

    if (path.takeRight(4) == ".csv") {
      true
    } else {
      false
    }

  }

  def fileContainsColumns(data: DataFrame): Boolean = {
    val necessary_cols = Array("Month", "DayofMonth", "DayOfWeek","DepTime","CRSDepTime","ArrTime","CRSArrTime",
      "UniqueCarrier","CRSElapsedTime","ArrDelay","DepDelay","Origin","Dest","Distance","TaxiOut")
    val cols_df = data.columns
    for (col <- necessary_cols) {
      if (cols_df.contains(col) == false) {
        println("The file does not contain all the necessary colums. The column " + col + " is missing.")
        return false
      }
    }
    true
  }

}
