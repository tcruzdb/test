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
// MAGIC #### Demonstrate how a collection of data can be easily queried and modified idempotently using coarse grained transactions.
// MAGIC 
// MAGIC ##### 1. Batch writes
// MAGIC ##### 2. Stream writes
// MAGIC ##### 3. Simulate write transaction failures
// MAGIC ##### 4. Access via `SQL`
// MAGIC ##### 5. Directories and Files
// MAGIC ##### 6. Joins and Upserts

// COMMAND ----------

import com.databricks.sql.transaction.tahoe._

case class Event(
  date: java.sql.Date, 
  eventType: String, 
  location: String,
  deviceId: Long
)

def createEvents(count: Int, date: String, eventType: String) = 
  (1 to count).toList.map(id => Event(java.sql.Date.valueOf(date), eventType, "NY", (id % 5) + 1))

val deltaDir = "/mnt/tcruz/delta/events" 
dbutils.fs.rm(deltaDir, true)

// COMMAND ----------

createEvents(1000, "2017-10-1", "batch-insert")
  .toDF()
  .write
  .format("delta")
  .save(deltaDir)

// COMMAND ----------

display(dbutils.fs.ls(deltaDir))

// COMMAND ----------

// display(spark.read.parquet("/mnt/tcruz/delta/events/part-00000-cad2b04e-484e-4250-a27b-ba57d79e7b81-c000.snappy.parquet"))

// COMMAND ----------

// reading from dbfs deltaDir; updates as underlying files change
display(
  spark
  .readStream
  .format("delta")
  .load(deltaDir)
  .groupBy("date", "eventType")
  .count
  .orderBy("date")
)

// COMMAND ----------

createEvents(5000, "2017-10-5", "batch-append")
  .toDF()
  .write
  .format("delta")
  .mode("append")
  .save(deltaDir)

// see streaming query update with events from 10/5

// COMMAND ----------

val df = createEvents(10000, "2017-10-10", "stream").toDF()
val schema = df.schema

df.write
  .format("json")
  .mode("overwrite")
  .save("/mnt/tcruz/eventsStream")

val streamDF = spark
  .readStream
  // specify schema
  .schema(schema)
  .option("maxFilesPerTrigger", 1)
  .json("/mnt/tcruz/eventsStream")

// COMMAND ----------

streamDF
  .writeStream
  .format("delta")
  .option("path", deltaDir)
  .option("checkpointLocation", "/tmp/tcruz/hg/checkpoint/")
  .start()

// see streaming query update with events from 10/10

// COMMAND ----------

// create events with bad date
val badDF = (
  (1 to 15000).toList.map(_ => ("2017-31-10", "batch-tx", "NY")) ++ 
  (1 to 15000).toList.map(_ => ("2017-10-31", "batch-tx", "NY"))
).toDF("date", "eventType", "location")

// COMMAND ----------

badDF
  .write
  .format("delta")
  .mode("append")
  .save(deltaDir)

// COMMAND ----------

import org.apache.spark.sql.functions.{coalesce, to_date}

val goodDF = 
badDF.withColumn(
  "date", 
  coalesce($"date".cast("date"), to_date($"date", "yyyy-dd-MM")))

// COMMAND ----------

goodDF
  .write
  .format("delta")
  .mode("append")
  .save(deltaDir)

// COMMAND ----------

// MAGIC %md 
// MAGIC #### [Delta QuickStart w/ SQL examples](https://docs.databricks.com/delta/delta-intro.html#quick-start-example)

// COMMAND ----------

// MAGIC %sql 
// MAGIC SELECT count(*), date, eventType 
// MAGIC FROM delta.`/mnt/tcruz/delta/events` 
// MAGIC GROUP BY date, eventType ORDER BY date

// COMMAND ----------

// MAGIC %sql 
// MAGIC USE tcruz;
// MAGIC 
// MAGIC DROP TABLE IF EXISTS events_view;
// MAGIC 
// MAGIC CREATE TABLE events_view(
// MAGIC   date DATE, 
// MAGIC   eventType STRING,
// MAGIC   location STRING,
// MAGIC   deviceId BIGINT) 
// MAGIC USING Delta
// MAGIC OPTIONS (path "/mnt/tcruz/delta/events")

// COMMAND ----------

/*
%sql 
USE tcruz;
CREATE OR REPLACE VIEW events_view AS 
SELECT * FROM delta.`/mnt/tcruz/delta/events`
*/

// COMMAND ----------

// MAGIC %sql 
// MAGIC SELECT count(1), date, eventType 
// MAGIC FROM events_view 
// MAGIC GROUP BY date, eventType ORDER BY date

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/delta/events/

// COMMAND ----------

dbutils.fs.ls("/mnt/tcruz/delta/events").size

// COMMAND ----------

// display(spark.read.parquet("/mnt/tcruz/delta/events/part-00027-e1ec548b-0388-4e74-84e8-6ddd1c0f629a-c000.snappy.parquet"))

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/delta/events/_delta_log/

// COMMAND ----------

// MAGIC %fs head /mnt/tcruz/delta/events/_delta_log/00000000000000000000.json

// COMMAND ----------

// MAGIC %sh 
// MAGIC head -2 /dbfs/mnt/tcruz/delta/events/_delta_log/00000000000000000000.json|tail -1|python -m json.tool

// COMMAND ----------

// MAGIC %sh head -1 /dbfs/mnt/tcruz/delta/events/_delta_log/00000000000000000001.json|python -m json.tool

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/delta/events/_delta_log/

// COMMAND ----------

// MAGIC %fs head /mnt/tcruz/delta/events/_delta_log/00000000000000000012.json

// COMMAND ----------

// display(spark.read.parquet("/mnt/tcruz/delta/events/_delta_log/00000000000000000030.checkpoint.parquet"))

// COMMAND ----------

// MAGIC %sql 
// MAGIC OPTIMIZE '/mnt/tcruz/delta/events'

// COMMAND ----------

// MAGIC %sql VACUUM '/mnt/tcruz/delta/events'

// COMMAND ----------

case class Device(
  deviceId: Long,
  deviceType: String
)

(1 to 5).toList.map(id => Device(id, s"device_$id"))
  .toDF()
  .write
  .format("parquet")
  .mode("overwrite")
  .saveAsTable("tcruz.devices")

// COMMAND ----------

// MAGIC %sql SELECT * FROM tcruz.devices

// COMMAND ----------

// MAGIC %sql 
// MAGIC USE tcruz;
// MAGIC 
// MAGIC SELECT deviceType, count(1) as count
// MAGIC FROM events_view e
// MAGIC JOIN devices d
// MAGIC ON e.deviceId = d.deviceId
// MAGIC GROUP BY deviceType ORDER BY deviceType

// COMMAND ----------

(1 to 10)
  .toList.map(id => Event(java.sql.Date.valueOf("2017-10-1"), "upsert", "CA", id))
  .toDF
  .createOrReplaceTempView("temp_table")

// COMMAND ----------

// MAGIC %sql SELECT * FROM temp_table

// COMMAND ----------

// MAGIC %sql SELECT count(*), location FROM delta.`/mnt/tcruz/delta/events` GROUP BY location

// COMMAND ----------

// MAGIC %sql
// MAGIC MERGE INTO delta.`/mnt/tcruz/delta/events` AS target
// MAGIC USING temp_table
// MAGIC ON temp_table.deviceId = target.deviceId AND temp_table.date = target.date
// MAGIC 
// MAGIC WHEN MATCHED THEN 
// MAGIC   UPDATE SET 
// MAGIC   target.date = temp_table.date, 
// MAGIC   target.eventType = temp_table.eventType, 
// MAGIC   target.location = temp_table.location, 
// MAGIC   target.deviceId = temp_table.deviceId
// MAGIC 
// MAGIC -- right-join
// MAGIC WHEN NOT MATCHED THEN 
// MAGIC   INSERT (target.date, target.eventType, target.location, target.deviceId) 
// MAGIC   VALUES (temp_table.date, temp_table.eventType, temp_table.location, temp_table.deviceId)

// COMMAND ----------

// MAGIC %sql 
// MAGIC SELECT count(*), location 
// MAGIC FROM delta.`/mnt/tcruz/delta/events` 
// MAGIC GROUP BY location

// COMMAND ----------

// MAGIC %md
// MAGIC <img alt="delta" src="https://databricks.com/wp-content/uploads/2017/10/Screen-Shot-2017-10-25-at-01.15.08.png" width="800"/>
// MAGIC 
// MAGIC Reference:
// MAGIC https://databricks.com/blog/2017/10/25/databricks-delta-a-unified-management-system-for-real-time-big-data.html

// COMMAND ----------

