# Databricks notebook source
symbols = ['AAPL', 'GOOGL', 'FB']

positions = map(lambda s: (s, 10000), symbols)
positions_df = spark.createDataFrame(positions).toDF('symbol', 'amount')

positions_df.coalesce(1).write.format("delta").mode("overwrite").save('/mnt/tcruz/positions/')

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists tcruz.positions 
# MAGIC using delta
# MAGIC location '/mnt/tcruz/positions'

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from tcruz.positions
# MAGIC -- alternate query, directly from delta path
# MAGIC -- select * from delta.`/mnt/tcruz/positions`

# COMMAND ----------

# MAGIC %sql describe extended tcruz.positions

# COMMAND ----------

# MAGIC %fs ls /mnt/tcruz/positions

# COMMAND ----------

import random
from pyspark.sql.functions import rand

new_symbols = ['MSFT', 'AMZN']

trades = []

# existing positions
for i in range(20000):
  symbol = symbols[random.randint(0, len(symbols)-1)]
  trades.append((symbol, random.randint(-90, 100)))

# new positions
for i in range(4000):
  symbol = new_symbols[random.randint(0, len(new_symbols)-1)]
  trades.append((symbol, random.randint(-20, 50)))

# shuffle trades
trades_df = spark.createDataFrame(trades).toDF('symbol', 'amount').orderBy(rand())

# COMMAND ----------

trades_df.createOrReplaceTempView("trades")
display(sql("select symbol, sum(amount) from trades group by 1 order by 2 desc"))

# COMMAND ----------

# run once, we'll simulate stream of trades
# trades_df.repartition(300).write.mode('overwrite').json("/mnt/tcruz/trades/")

# COMMAND ----------

# what's a good value for this?
spark.conf.set("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

# clean out the checkpoint directory
dbutils.fs.rm("/mnt/tcruz/checkpoint-positions", True)

# COMMAND ----------

from pyspark.sql.types import *

schema = StructType([ \
    StructField("symbol", StringType(), False), \
    StructField("amount", LongType(), False)])

# COMMAND ----------

(spark
  .readStream
  .schema(schema)
  .option("maxFilesPerTrigger", 1)
  .json("/mnt/tcruz/trades/")
  .writeStream
  .format("delta")
  .option("checkpointLocation", "/mnt/tcruz/checkpoint-positions")
  .option('path', '/mnt/tcruz/positions')
  .outputMode("append")
  .start())

# COMMAND ----------

streamingAggregation = (
  spark
    .readStream
    .format("delta")
    .load("/mnt/tcruz/positions/")
    .groupBy("symbol")
    .sum("amount")
    .orderBy("symbol")
)
display(streamingAggregation)

# COMMAND ----------

display(sql("select symbol, sum(amount) from tcruz.positions group by symbol order by symbol"))

# COMMAND ----------

# MAGIC %sh ls /dbfs/mnt/tcruz/

# COMMAND ----------

# MAGIC %sh ls /dbfs/mnt/tcruz/checkpoint-positions/offsets/

# COMMAND ----------

# MAGIC %sh cat /dbfs/mnt/tcruz/checkpoint-positions/offsets/10

# COMMAND ----------

# MAGIC %sh tail -2 /dbfs/mnt/tcruz/checkpoint-positions/offsets/10|head -1|python -m json.tool

# COMMAND ----------

# MAGIC %sh cat /dbfs/mnt/tcruz/positions/_delta_log/00000000000000000001.json

# COMMAND ----------

# MAGIC %sh head -2 /dbfs/mnt/tcruz/positions/_delta_log/00000000000000000445.json|tail -1|python -mjson.tool

# COMMAND ----------

