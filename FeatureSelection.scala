package upm.bd

import org.apache.spark.ml.feature.{UnivariateFeatureSelector, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{monotonicallyIncreasingId, monotonically_increasing_id}

object FeatureSelection {

  def featureSelection( data_train: DataFrame,  data_test: DataFrame): Array[DataFrame]= {

    //var data2 = data.select("TaxiOut","DayOfWeek", "ArrDelay", "MonthSine", "MonthCos", "DestLong", "DestLat", "CRSArrTimeMin","DepDelay")

    data_train.show()

    var cols = data_train.drop("ArrDelay").columns

    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")

    var features_train = assembler.transform(data_train)
    features_train = features_train.select("features", "ArrDelay")

    val selector = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous")
      .setSelectionMode("fwe")
      .setSelectionThreshold(0.05)
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    val feat_selected = selector.fit(features_train.select("features", "ArrDelay")).selectedFeatures
    println(s"Index of selected features ${feat_selected.mkString(" ")}")

    
    
    var train = data_train.drop("ArrDelay") 
    var test = data_test.drop("ArrDelay")
    val colslen = train.columns.length

    for (i <- 0 to colslen-1) {
      if (!feat_selected.contains(i)) {
        train = train.drop(data_train.columns(i))
        test = test.drop(data_test.columns(i))   
      }
    }

    train = train.withColumn("id", monotonically_increasing_id)
    test = test.withColumn("id", monotonically_increasing_id)


    train = train.join(data_train.withColumn("id", monotonically_increasing_id).select("id", "ArrDelay"), Seq("id"), "left").drop("id")
    test = test.join(data_test.withColumn("id", monotonically_increasing_id).select("id", "ArrDelay"), Seq("id"), "left").drop("id")
    train.show()
    Array(train, test)
  }

}
