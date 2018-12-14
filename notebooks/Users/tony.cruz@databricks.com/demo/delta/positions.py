# Databricks notebook source
symbols = ['AAPL', 'GOOGL', 'FB', 'MSFT', 'AMZN']
positions = map(lambda s: (s, 10000), symbols)
positions_df = spark.createDataFrame(positions).toDF('symbol', 'amount')

# COMMAND ----------

display(positions_df)

# COMMAND ----------

positions_df.rdd.getNumPartitions()

# COMMAND ----------

positions_df.coalesce(1).write.format("delta").mode("overwrite").save('/mnt/tcruz/positions/')

# COMMAND ----------

# MAGIC %sql select * from delta.`/mnt/tcruz/positions`

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists tcruz.positions 
# MAGIC using delta
# MAGIC location '/mnt/tcruz/positions'

# COMMAND ----------

# MAGIC %sql select * from tcruz.positions

# COMMAND ----------

# MAGIC %sql describe extended tcruz.positions

# COMMAND ----------

# MAGIC %fs ls /mnt/tcruz/positions

# COMMAND ----------

import random
from pyspark.sql.functions import rand

new_symbols = ['CRM', 'NFLX', 'INTU', 'SHOP']

trades = []

# trade existing positions (MERGE)
for i in range(10000):
  symbol = symbols[random.randint(0, len(symbols)-1)]
  trades.append((symbol, random.randint(-100, 100)))

# add new positions (INSERT)
for i in range(2000):
  symbol = new_symbols[random.randint(0, len(new_symbols)-1)]
  trades.append((symbol, random.randint(1, 50)))

trades_df = spark.createDataFrame(trades).toDF('symbol', 'delta').orderBy(rand())

# COMMAND ----------

display(trades_df)

# COMMAND ----------

# run once, we'll simulate stream of trades
# trades_df.repartition(500).write.mode('overwrite').json("/mnt/tcruz/trades/")

# COMMAND ----------

# MAGIC %fs ls /mnt/tcruz/trades

# COMMAND ----------

inputStream = spark \
    .readStream \
    .schema(trades_df.schema) \
    .option("maxFilesPerTrigger", 1) \
    .json("/mnt/tcruz/trades/") \
    .createOrReplaceTempView("trades")

# COMMAND ----------

# MAGIC %sql select symbol, sum(delta) from trades group by symbol

# COMMAND ----------

