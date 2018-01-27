// Databricks notebook source
// MAGIC %md #### Click Prediction
// MAGIC ####[Ad impressions with clicks dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data)
// MAGIC 
// MAGIC <img src="/files/img/fraud_ml_pipeline.png" alt="workflow" width="500">
// MAGIC 
// MAGIC #### 1. ETL
// MAGIC #### 2. Data Exploration / SQL
// MAGIC #### 3. Advanced Analytics / ML

// COMMAND ----------

// MAGIC %sh ls -lh /dbfs/mnt/adtech/impression/csv/train.csv/

// COMMAND ----------

// MAGIC %fs head /mnt/adtech/impression/csv/train.csv/part-00000-tid-695242982481829702-ed48f068-bfdf-484a-a397-c4144e4897d8-0-c000.csv

// COMMAND ----------

// MAGIC %fs ls /mnt/adtech/impression/

// COMMAND ----------

val df = spark.read
  .option("header", true)
  .option("inferSchema", true)
  .csv("/mnt/adtech/impression/csv/train.csv/")

// COMMAND ----------

df.count

// COMMAND ----------

df.printSchema

// COMMAND ----------

df.coalesce(4)
  .write
  .mode("overwrite")
  .parquet("/mnt/adtech/impression/parquet/train.csv")

// COMMAND ----------

// MAGIC %fs ls /mnt/adtech/impression/parquet/train.csv/

// COMMAND ----------

// MAGIC %sh ls -lh /dbfs/mnt/adtech/impression/parquet/train.csv/