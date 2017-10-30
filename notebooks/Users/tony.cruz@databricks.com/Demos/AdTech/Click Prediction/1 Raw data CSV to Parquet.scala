// Databricks notebook source
// MAGIC %md ## Click Prediction
// MAGIC https://www.kaggle.com/c/avazu-ctr-prediction/data
// MAGIC 
// MAGIC <img src="/files/img/fraud_ml_pipeline.png" alt="workflow" width="500">
// MAGIC 
// MAGIC #### 1. Load raw data
// MAGIC #### 2. Explore data, features
// MAGIC #### 3. Create ML model
// MAGIC #### 4. Evaluate

// COMMAND ----------

// MAGIC %fs ls /mnt/adtech-demo/original-gz-files/

// COMMAND ----------

val total: Long = dbutils.fs.ls("/mnt/adtech-demo/original-gz-files/train.gz").map(f => f.size).sum
println(s"$total bytes")
println(s"${total/1024.0/1024.0/1024.0} GB")

// COMMAND ----------

// MAGIC %fs head /mnt/adtech-demo/uncompressed/train.csv

// COMMAND ----------

val total: Long = dbutils.fs.ls("/mnt/adtech-demo/uncompressed/train.csv").map(f => f.size).sum
println(s"$total bytes")
println(s"${total/Math.pow(1024.0, 3)} GB")

// COMMAND ----------

// MAGIC %fs head /mnt/adtech-demo/uncompressed/test.csv

// COMMAND ----------

val dfRaw = spark.read
  .option("header", true)
  .option("inferSchema", true)
  .csv("/mnt/adtech-demo/uncompressed/train.csv")

// COMMAND ----------

dfRaw.count

// COMMAND ----------

dfRaw.printSchema

// COMMAND ----------

import org.apache.spark.sql.types._
val schema = StructType(
  Array(
    StructField("id", StringType, false),
    
    StructField("click", IntegerType, true),
    StructField("hour", IntegerType, true),
    StructField("C1", IntegerType, true),
    StructField("banner_pos", IntegerType, true),
    StructField("site_id", StringType, true),
    StructField("site_domain", StringType, true),
    StructField("site_category", StringType, true),
    StructField("app_id", StringType, true),
    StructField("app_domain", StringType, true),
    StructField("app_category", StringType, true),
    StructField("device_id", StringType, true),
    StructField("device_ip", StringType, true),
    StructField("device_model", StringType, true),
    StructField("device_type", IntegerType, true),
    StructField("device_conn_type", IntegerType, true),
    StructField("C14", IntegerType, true),
    StructField("C15", IntegerType, true),
    StructField("C16", IntegerType, true),
    StructField("C17", IntegerType, true),
    StructField("C18", IntegerType, true),
    StructField("C19", IntegerType, true),
    StructField("C20", IntegerType, true),
    StructField("C21", IntegerType, true)
  )
)

// COMMAND ----------

val df = spark.read
  .option("header", true)
  .schema(schema)
  .csv("/mnt/adtech-demo/uncompressed/train.csv")
df.count

// COMMAND ----------

// Please update your clone of this demo to write parquet files to your own directory

// df.coalesce(4).write.mode("overwrite").parquet("/mnt/tcruz/adtech-demo/parquet/impression")

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/adtech-demo/parquet/impression

// COMMAND ----------

val total: Long = dbutils.fs.ls("/mnt/tcruz/adtech-demo/parquet/impression").map(f => f.size).sum
println(s"$total bytes")
println(s"${total/1024.0/1024.0/1024.0} GB")

// COMMAND ----------

