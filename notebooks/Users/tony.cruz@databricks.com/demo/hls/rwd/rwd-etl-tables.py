# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://assets.cms.gov/resources/cms/images/logo/site-logo.png" />
# MAGIC 
# MAGIC ### [CMS Dataset Downloads](https://www.cms.gov/OpenPayments/Explore-the-Data/Dataset-Downloads.html): General Payments 2016
# MAGIC 
# MAGIC * `OP_DTL_GNRL_PGYR2016_P01172018.csv: 
# MAGIC This file contains the data set of General Payments reported for the 2016 program year.`

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/pub-tc/img-fraud_ml_pipeline.png" alt="workflow" width="500">

# COMMAND ----------

# MAGIC %sh 
# MAGIC wget http://download.cms.gov/openpayments/PGYR16_P011718.ZIP
# MAGIC unzip PGYR16_P011718.ZIP
# MAGIC ls -lh

# COMMAND ----------

# MAGIC %sh 
# MAGIC gzip OP_DTL_GNRL_PGYR2016_P01172018.csv

# COMMAND ----------

# MAGIC %sh cp OP_DTL_GNRL_PGYR2016_P01172018.csv.gz /dbfs/mnt/tcruz/hls/cms/

# COMMAND ----------

# MAGIC %fs ls /mnt/tcruz/hls/cms/

# COMMAND ----------

# MAGIC %sh zcat /dbfs/mnt/tcruz/hls/cms/OP_DTL_GNRL_PGYR2016_P01172018.csv.gz|head -10

# COMMAND ----------

cms2016 = spark.read \
  .option('header', 'true') \
  .option('inferSchema', 'true') \
  .csv('/mnt/tcruz/hls/cms/OP_DTL_GNRL_PGYR2016_P01172018.csv.gz') \
  .repartition(4)

# COMMAND ----------

cms2016.printSchema()

# COMMAND ----------

cms2016.count()

# COMMAND ----------

# MAGIC %sql create database if not exists cms

# COMMAND ----------

cms2016.coalesce(4) \
  .write \
  .format('parquet') \
  .mode('overwrite') \
  .saveAsTable('cms.gp2016')

# COMMAND ----------

# MAGIC %sql 
# MAGIC use cms;
# MAGIC 
# MAGIC select length(recipient_state), count(1)
# MAGIC from gp2016
# MAGIC group by 1 
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct recipient_state)
# MAGIC from cms.gp2016
# MAGIC where length(recipient_state) = 2

# COMMAND ----------

states = spark.read \
  .jdbc("jdbc:mysql://databricks-customer-example.chlee7xx28jo.us-west-2.rds.amazonaws.com:3306?user=root&password=tMFqirgjz1lc", "databricks.fmstates") \
  .select('state_abbr') \
  .distinct() \
  .orderBy('state_abbr')

# COMMAND ----------

display(states)

# COMMAND ----------

cmsgeo = cms2016.join(states, states.state_abbr == cms2016.Recipient_State)

# COMMAND ----------

cmsgeo.coalesce(4) \
  .write \
  .format('parquet') \
  .saveAsTable('cms.gp2016geo')

# COMMAND ----------

print(cms2016.count())
print(cmsgeo.count())
print(cms2016.count() - cmsgeo.count())