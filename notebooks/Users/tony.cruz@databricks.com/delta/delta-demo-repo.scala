// Databricks notebook source
// MAGIC %md
// MAGIC <img alt="before" src="https://s3.us-east-2.amazonaws.com/databricks-roy/before_delta.jpeg" width="800">

// COMMAND ----------

// MAGIC %md
// MAGIC <img alt="before" src="https://s3.us-east-2.amazonaws.com/databricks-roy/after_delta.jpeg" width="800">

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### [Databricks Delta](https://docs.databricks.com/delta/index.html)
// MAGIC 
// MAGIC ##### Transactional storage layer over cloud storage systems (e.g. S3) supported by DBFS
// MAGIC 
// MAGIC ##### Delta 
// MAGIC   * stores Parquet files in DBFS
// MAGIC   * maintains a transaction log that efficiently and correctly tracks which files are currently part of the table
// MAGIC 
// MAGIC ##### Features
// MAGIC * **`ACID`** Transactions - Multiple writers from different clusters can safely modify the same table without fear of corruption or inconsistency.
// MAGIC * **`DELETES/UPDATES/UPSERTS`** - Transactionally modify existing data without disturbing jobs that are reading the same dataset.
// MAGIC * **Data Validation** - Ensure that all data meets specified invariants (for example, `NOT NULL`) by rejecting invalid data.
// MAGIC * **Automatic File Management** - Automatic support for organizing data into large files that can be read efficiently.
// MAGIC * **Statistics and Data Skipping** - Track statistics about the data in each file and will avoid reading irrelevant information, speeding-up reads by 10-100x.

// COMMAND ----------

// MAGIC %md
// MAGIC <img alt="delta" src="https://databricks.com/wp-content/uploads/2017/10/Screen-Shot-2017-10-25-at-01.15.08.png" width="850"/>
// MAGIC 
// MAGIC 
// MAGIC #### 1. Setup a Delta directory in DBFS with schema
// MAGIC 
// MAGIC #### 2. Create a table over Delta directory in the metastore
// MAGIC 
// MAGIC #### 3. Add files to our Delta directory
// MAGIC 
// MAGIC #### 4. Access our Delta data with `SQL`
// MAGIC 
// MAGIC #### 5. Delta under the hood -- the transaction log
// MAGIC 
// MAGIC #### 6. Upsert data into our Delta table
// MAGIC 
// MAGIC #### 7. `OPTIMIZE` and `VACUUM`

// COMMAND ----------

val deltaDir = "/mnt/tcruz/demo/repo/delta/test/"
dbutils.fs.rm(deltaDir, true)

// COMMAND ----------

import org.apache.spark.sql.functions.lit
val df = spark
  .range((1e6).toLong)
  .repartition(500)
  .withColumn("event_type", lit("batch-insert-initial"))

// COMMAND ----------

display(df)

// COMMAND ----------

/* 
We're not writing any data here.
We're specifying that:
1. Delta is managing this directory '/mnt/tcruz/demo/repo/delta/test/'
2. Files in this directory have the schema of this dataframe
*/
df.limit(0)
  .write
  .format("delta")
  .mode("append")
  .save(deltaDir)

// COMMAND ----------

// We'll talk about these files a little further down
display(dbutils.fs.ls(deltaDir))

// COMMAND ----------

// spark.read.parquet("/mnt/tcruz/demo/repo/delta/test/part-00000-65a215ff-12f4-4f51-a634-c966d535260e-c000.snappy.parquet").count

// COMMAND ----------

// MAGIC %sql
// MAGIC -- we create a table in the metastore with the path option to the delta directory
// MAGIC USE tcruz;
// MAGIC 
// MAGIC DROP TABLE IF EXISTS delta_table;
// MAGIC 
// MAGIC CREATE TABLE delta_table(
// MAGIC   id bigint, 
// MAGIC   event_type string) 
// MAGIC USING Delta
// MAGIC OPTIONS (path "/mnt/tcruz/demo/repo/delta/test/")

// COMMAND ----------

// MAGIC %sql describe extended delta_table

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * FROM delta_table

// COMMAND ----------

// MAGIC %sql
// MAGIC -- can also query Delta directory directly
// MAGIC select * from delta.`/mnt/tcruz/demo/repo/delta/test`

// COMMAND ----------

// run once
df.write
  .format("delta")
  .mode("append")
  .save(deltaDir)

// COMMAND ----------

// MAGIC %sql SELECT count(*) FROM delta_table

// COMMAND ----------

// MAGIC %sql select * from delta_table order by id desc

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/demo/repo/delta/test/

// COMMAND ----------

dbutils.fs.ls("/mnt/tcruz/demo/repo/delta/test/").size

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/demo/repo/delta/test/_delta_log/

// COMMAND ----------

// MAGIC %fs head /mnt/tcruz/demo/repo/delta/test/_delta_log/00000000000000000000.json

// COMMAND ----------

// MAGIC %sh 
// MAGIC # let's look at the metadata for this delta directory
// MAGIC 
// MAGIC head -2 /dbfs/mnt/tcruz/demo/repo/delta/test/_delta_log/00000000000000000000.json|tail -1|python -m json.tool

// COMMAND ----------

// MAGIC %fs head /mnt/tcruz/demo/repo/delta/test/_delta_log/00000000000000000001.json

// COMMAND ----------

// MAGIC %sh head -1 /dbfs/mnt/tcruz/demo/repo/delta/test/_delta_log/00000000000000000001.json|python -m json.tool

// COMMAND ----------

// create a new table with upsert data with same schema and overlapping ids
spark.range(500)
  .coalesce(100)
  .withColumn("event_type", lit("batch-upsert"))
  .where("id % 2 = 0")
  .write
  .mode("overwrite")
  .saveAsTable("delta_upsert")

// COMMAND ----------

// MAGIC %sql select * from delta_upsert order by id

// COMMAND ----------

// MAGIC %sql select count(*) from delta_upsert

// COMMAND ----------

// MAGIC %sql
// MAGIC -- here we merge new data into delta_table
// MAGIC MERGE INTO delta_table AS target
// MAGIC USING delta_upsert AS source
// MAGIC ON source.id = target.id 
// MAGIC 
// MAGIC WHEN MATCHED 
// MAGIC   THEN UPDATE SET target.id = source.id, target.event_type = source.event_type
// MAGIC 
// MAGIC -- right-join
// MAGIC WHEN NOT MATCHED 
// MAGIC   THEN INSERT (target.id, target.event_type) VALUES (source.id, source.event_type)

// COMMAND ----------

// MAGIC %sql select * from delta_table where id < 500 order by id

// COMMAND ----------

// MAGIC %sql select * from delta_table where id >= 500 order by id

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/demo/repo/delta/test/_delta_log/

// COMMAND ----------

// MAGIC %fs head /mnt/tcruz/demo/repo/delta/test/_delta_log/00000000000000000002.json

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/demo/repo/delta/test

// COMMAND ----------

dbutils.fs.ls(deltaDir).size

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/demo/repo/delta/test/_delta_log

// COMMAND ----------

// MAGIC %fs head /mnt/tcruz/demo/repo/delta/test/_delta_log/00000000000000000002.json

// COMMAND ----------

// MAGIC %sql OPTIMIZE '/mnt/tcruz/demo/repo/delta/test/'

// COMMAND ----------

dbutils.fs.ls(deltaDir).size

// COMMAND ----------

// sql(""" OPTIMIZE '/mnt/tcruz/demo/repo/delta/test/' """)

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/demo/repo/delta/test/_delta_log

// COMMAND ----------

// MAGIC %fs head /mnt/tcruz/demo/repo/delta/test/_delta_log/00000000000000000003.json

// COMMAND ----------

// MAGIC %sql 
// MAGIC -- this should remove the files in delta dir that are not part of the table
// MAGIC VACUUM '/mnt/tcruz/demo/repo/delta/test/'

// COMMAND ----------

dbutils.fs.ls(deltaDir).size

// COMMAND ----------

// MAGIC %sql DROP TABLE IF EXISTS delta_upsert

// COMMAND ----------

