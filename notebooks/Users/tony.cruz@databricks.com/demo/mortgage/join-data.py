# Databricks notebook source
'''states = spark.read.jdbc("jdbc:mysql://databricks-customer-example.chlee7xx28jo.us-west-2.rds.amazonaws.com:3306?user=root&password=tMFqirgjz1lc", "databricks.fmstates")'''

# COMMAND ----------

# MAGIC %sh cat /dbfs/FileStore/state_codes.csv

# COMMAND ----------

states = spark.read.option('header', False).csv('/FileStore/state_codes.csv').withColumnRenamed('_c0', 'usps_code')

# COMMAND ----------

display(states)

# COMMAND ----------

df = table("mortgage")

# COMMAND ----------

spark.catalog.dropGlobalTempView("mortgages_with_geolocation")
spark.conf.set("spark.sql.shuffle.partitions", 16)
withGeo = df.join(states, "usps_code")
withGeo.createGlobalTempView("mortgages_with_geolocation")

# COMMAND ----------

withGeo.printSchema()

# COMMAND ----------

# MAGIC %sql cache table global_temp.mortgages_with_geolocation

# COMMAND ----------

# MAGIC %sql select state_abbr, avg(unpaid_balance) as unpaid 
# MAGIC from global_temp.mortgages_with_geolocation  
# MAGIC where year ="$year" 
# MAGIC group by state_abbr

# COMMAND ----------

# MAGIC %sql select state_abbr, avg(unpaid_balance) as unpaid 
# MAGIC from global_temp.mortgages_with_geolocation  
# MAGIC where year ="$year" group by state_abbr

# COMMAND ----------

# MAGIC %sql 
# MAGIC select 
# MAGIC   state_abbr, 
# MAGIC   year, 
# MAGIC   avg(unpaid_balance), 
# MAGIC   avg(unpaid_balance) - lag(avg(unpaid_balance), 1, null) over (partition by state_abbr order by year) as delta
# MAGIC from global_temp.mortgages_with_geolocation
# MAGIC where state_abbr in ('CA', 'NY')
# MAGIC group by 1,2
# MAGIC order by state_abbr, year

# COMMAND ----------

# MAGIC %sql 
# MAGIC select 
# MAGIC   state_abbr, 
# MAGIC   year, 
# MAGIC   avg(unpaid_balance), 
# MAGIC   avg(unpaid_balance) - lag(avg(unpaid_balance), 1, null) over (partition by state_abbr order by year) as delta
# MAGIC from global_temp.mortgages_with_geolocation
# MAGIC where state_abbr = 'CA'
# MAGIC group by 1,2
# MAGIC order by state_abbr, year

# COMMAND ----------

