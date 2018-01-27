# Databricks notebook source
# MAGIC %md
# MAGIC #### 2. Data Exploration / SQL

# COMMAND ----------

impression = spark.read.parquet("/mnt/adtech/impression/parquet/train.csv/")
impression.createOrReplaceTempView("impression")

# COMMAND ----------

# MAGIC %sql describe impression

# COMMAND ----------

# MAGIC %sql select banner_pos, count(1)
# MAGIC from impression
# MAGIC group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select banner_pos,
# MAGIC sum(case when click = 1 then 1 else 0 end) as click,
# MAGIC sum(case when click = 0 then 1 else 0 end) as no_click
# MAGIC from impression group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select banner_pos,
# MAGIC sum(case when click = 1 then 1 else 0 end) / (count(1) * 1.0) as CTR
# MAGIC from impression group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select device_type, count(1)
# MAGIC from impression
# MAGIC group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select device_type,
# MAGIC sum(case when click = 1 then 1 else 0 end) as click,
# MAGIC sum(case when click = 0 then 1 else 0 end) as no_click
# MAGIC from impression group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select device_type,
# MAGIC sum(case when click = 1 then 1 else 0 end) / (count(1) * 1.0) as CTR
# MAGIC from impression group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select site_category, count(1) as count
# MAGIC from impression
# MAGIC group by 1 having count > 200 order by count desc

# COMMAND ----------

# MAGIC %sql select site_category,
# MAGIC sum(case when click = 1 then 1 else 0 end) as click,
# MAGIC sum(case when click = 0 then 1 else 0 end) as no_click
# MAGIC from impression group by 1 order by 3 desc

# COMMAND ----------

# MAGIC %sql select site_category,
# MAGIC sum(case when click = 1 then 1 else 0 end) / (count(1) * 1.0) as CTR
# MAGIC from impression group by 1 order by 2 desc

# COMMAND ----------

# MAGIC %sql select substr(hour, 7) as hour, 
# MAGIC count(1)
# MAGIC from impression 
# MAGIC group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select substr(hour, 7) as hour,
# MAGIC sum(case when click = 1 then 1 else 0 end) as click,
# MAGIC sum(case when click = 0 then 1 else 0 end) as no_click
# MAGIC from impression group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select substr(hour, 7) as hour,
# MAGIC sum(case when click = 1 then 1 else 0 end) / (count(1) * 1.0) as CTR
# MAGIC from impression group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select 
# MAGIC count(1) as total,
# MAGIC 
# MAGIC count(distinct C1) as C1,
# MAGIC count(distinct banner_pos) as banner_pos,
# MAGIC count(distinct site_id) as site_id,
# MAGIC count(distinct site_domain) as site_domain,
# MAGIC count(distinct site_category) as site_category,
# MAGIC count(distinct app_id) as app_id,
# MAGIC count(distinct app_domain) as app_domain,
# MAGIC count(distinct app_category) as app_category,
# MAGIC count(distinct device_id) as device_id,
# MAGIC count(distinct device_ip) as device_ip,
# MAGIC count(distinct device_model) as device_model,
# MAGIC count(distinct device_type) as device_type,
# MAGIC count(distinct device_conn_type) as device_conn_type,
# MAGIC count(distinct C14) as C14,
# MAGIC count(distinct C15) as C15,
# MAGIC count(distinct C16) as C16,
# MAGIC count(distinct C17) as C17,
# MAGIC count(distinct C18) as C18,
# MAGIC count(distinct C19) as C19,
# MAGIC count(distinct C20) as C20,
# MAGIC count(distinct C21) as C21
# MAGIC from impression

# COMMAND ----------

display(impression.describe())