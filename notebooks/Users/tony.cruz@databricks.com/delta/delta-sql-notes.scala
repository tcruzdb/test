// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC * create table with column not null throws exception
// MAGIC * query on delta table is not streaming
// MAGIC * no table() on DataStreamWriter/DataFrameWriter
// MAGIC * location/eventType column issue cmd 27

// COMMAND ----------

/*
SSEU Demo:

CREATE TABLE events(
  date DATE NOT NULL, // throws Exception
  eventType STRING,
  city STRING
)
USING delta

-------------------
spark.readStream.kafka(topic="events")
  .select($"value")
  .writeStream
  .table("events") // no table() in DataStreamWriter?

-------------------
// not a streaming query
SELECT COUNT(*), date
FROM events
GROUP BY date
ORDER BY date

-------------------
spark.read.json("/mnt/historical")
  .write
  .table("events") // no table() in DataFrameWriter?
  
-------------------
spark.read.json("/mnt/historical")
  .withColumn("date", coalesce($"date".cast("date"), to_date($"date", "MM-dd-yyyy")))
  .write
  .table("events") // no table()?
  
-------------------
sql("SELECT * FROM events WHERE city = 'Dublin'").first
// returns json with stats

*/


// COMMAND ----------

// basedir
val baseDir = "/mnt/tcruz/delta/sql/"

// delta parquet files for delta events table
val dbfsDeltaDir = baseDir + "events"

// historical event data
val dbfsHisDir = baseDir + "historical"

// streaming data
val dbfsStrDir = baseDir + "streaming"

// COMMAND ----------

// MAGIC %sql
// MAGIC use tcruz;
// MAGIC drop table if exists events;
// MAGIC show tables

// COMMAND ----------

import com.databricks.sql.transaction.tahoe._

case class Event(
  date: java.sql.Date,
  eventType: String, 
  location: String,
  deviceId: Long
)

def createEvents(count: Int, 
                 date: java.sql.Date, 
                 eventType: String, 
                 location: String) = 
  (1 to count).toList.map(id => Event(
    date,
    eventType,
    location,
    (id % 5) + 1))

// COMMAND ----------

val sdf = new java.text.SimpleDateFormat("yyyy-MM-dd")
val d = sdf.parse("2017-12-31").getTime

// COMMAND ----------

dbfsHisDir

// COMMAND ----------

dbutils.fs.rm(dbfsHisDir, true)

val r = scala.util.Random
val sdf = new java.text.SimpleDateFormat("yyyy-MM-dd")

(1 to 30).foreach(i => {
  val dateStr = s"""2017-12-${"%02d".format(i)}"""
  val dateMs = sdf.parse(dateStr).getTime
  
  createEvents(i * 1000 + r.nextInt(i * 1000), 
               new java.sql.Date(dateMs), 
               "batch-insert",
               "NY")
  .toDF()
  .write
  .mode("overwrite")
  .json(dbfsHisDir + "/" + dateStr)
})

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/delta/sql/historical/

// COMMAND ----------

display(spark.read.json("/mnt/tcruz/delta/sql/historical/2017-12-01/"))

// COMMAND ----------

dbutils.fs.rm(dbfsStrDir, true)

val r = scala.util.Random
val sdf = new java.text.SimpleDateFormat("yyyy-MM-dd")

createEvents(31 * 1000 + r.nextInt(31 * 1000), 
             new java.sql.Date(sdf.parse("2017-12-31").getTime), 
             "streaming", 
             "CA")
.toDF()
.repartition(100) // 500
.write
.mode("overwrite")
.json(dbfsStrDir + "/2017-12-31")

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/delta/sql/streaming/2017-12-31/

// COMMAND ----------

display(spark.read.json("/mnt/tcruz/delta/sql/streaming/2017-12-31/"))

// COMMAND ----------

dbfsDeltaDir

// COMMAND ----------

// this is the location where we'll write the delta table data
dbutils.fs.rm(dbfsDeltaDir, true)

// COMMAND ----------

// MAGIC %sql 
// MAGIC USE tcruz;
// MAGIC 
// MAGIC DROP TABLE IF EXISTS events;
// MAGIC 
// MAGIC -- this directory does not exist at this point, this will create directory with schema
// MAGIC CREATE TABLE events(
// MAGIC   date DATE, -- NOT NULL throws exception
// MAGIC   eventType STRING,
// MAGIC   location STRING,
// MAGIC   deviceType INT
// MAGIC )
// MAGIC USING delta
// MAGIC -- LOCATION '/mnt/tcruz/delta/sql/events';
// MAGIC OPTIONS (path '/mnt/tcruz/delta/sql/events');

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/delta/sql/events/

// COMMAND ----------

// MAGIC %fs ls /mnt/tcruz/delta/sql/events/_delta_log

// COMMAND ----------

// MAGIC %fs head /mnt/tcruz/delta/sql/events/_delta_log/00000000000000000000.json

// COMMAND ----------

// MAGIC %sh 
// MAGIC # Delta metadata; notice schema from CREATE TABLE
// MAGIC head -2 /dbfs/mnt/tcruz/delta/sql/events/_delta_log/00000000000000000000.json|tail -1|python -m json.tool

// COMMAND ----------

// MAGIC %sql describe extended tcruz.events

// COMMAND ----------

// MAGIC %sql 
// MAGIC use tcruz;
// MAGIC select date, count(1) 
// MAGIC from events 
// MAGIC group by date order by date

// COMMAND ----------

import org.apache.spark.sql.functions._

spark
  .readStream
  .schema( spark.read.json(dbfsStrDir + "/2017-12-31").schema )
  .option("maxFilesPerTrigger", 1)
  .json(dbfsStrDir + "/2017-12-31")
  .withColumn("date", $"date".cast("date"))
  .withColumn("deviceId", $"deviceId".cast("int"))

  .writeStream
  .format("delta")
  .option("path", dbfsDeltaDir)
  .option("checkpointLocation", "/tmp/tcruz/delta/checkpoint/")
  .start()


// COMMAND ----------

// MAGIC %sql 
// MAGIC -- refresh table tcruz.events;
// MAGIC -- not a streaming query
// MAGIC select date, count(1) as count
// MAGIC from tcruz.events 
// MAGIC group by date order by date

// COMMAND ----------

spark.read.json(dbfsHisDir + "/*")
  .withColumn("date", coalesce($"date".cast("date"), to_date($"date", "yyyy-dd-MM")))
  .withColumn("deviceId", $"deviceId".cast("int"))
  .write
  .format("delta")
  .mode("append")
  .insertInto("`tcruz`.`events`")

// COMMAND ----------

sql("SELECT * FROM tcruz.events WHERE date = '2017-12-24'").first

// COMMAND ----------

// MAGIC %sql 
// MAGIC -- eventType is showing up as location? source json/dataframe looks fine
// MAGIC select location, count(1) 
// MAGIC from events 
// MAGIC group by 1 order by 2 desc

// COMMAND ----------

display(spark.read.json("/mnt/tcruz/delta/sql/historical/2017-12-01/"))

// COMMAND ----------

// MAGIC %sh head -10 /dbfs/mnt/tcruz/delta/sql/historical/2017-12-01/part-00000-*.json

// COMMAND ----------

// MAGIC %sql OPTIMIZE '/mnt/tcruz/delta/sql/events'

// COMMAND ----------

// MAGIC %sql VACUUM '/mnt/tcruz/delta/sql/events'

// COMMAND ----------

/*
import org.apache.spark.sql.types._

val schema = new StructType()
  .add(StructField("date", DateType, false))
  .add(StructField("eventType", StringType, false))
  .add(StructField("location", StringType, false))
  .add(StructField("deviceType", IntegerType, false))

val df = spark.read.schema(schema).json(dbfsHisDir + "/*")
*/

// COMMAND ----------

/*df.write
  .format("delta")
  .mode("overwrite")
  .save(dbfsDir)*/

// COMMAND ----------

/*df.write
  .format("delta")
  .mode("append")
  // .saveAsTable("`tcruz`.`events`")
  .insertInto("`tcruz`.`events`")*/