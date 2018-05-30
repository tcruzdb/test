# Databricks notebook source
# MAGIC %md
# MAGIC [Dataset Reference: https://www.fhfa.gov/DataTools/Downloads/Pages/Public-Use-Databases.aspx](https://www.fhfa.gov/DataTools/Downloads/Pages/Public-Use-Databases.aspx)
# MAGIC 
# MAGIC [Data Dictionary: https://www.fhfa.gov/DataTools/Downloads/Documents/Enterprise-PUDB/Single-Family_Census_Tract_File_/2016_Single_Family_Census_Tract_File.pdf](https://www.fhfa.gov/DataTools/Downloads/Documents/Enterprise-PUDB/Single-Family_Census_Tract_File_/2016_Single_Family_Census_Tract_File.pdf)

# COMMAND ----------

# MAGIC %fs ls /mnt/tcruz/mortgage_data

# COMMAND ----------

# MAGIC %sh ls -lR /dbfs/mnt/tcruz/mortgage_data/gz/

# COMMAND ----------

# MAGIC %fs head /mnt/mortgage_data/csv/2016/fnma_sf2016c_loans.csv/part-00000-tid-1402114478678342897-39e67c51-6226-4b93-84fe-e53eb758697a-0-c000.csv

# COMMAND ----------

# MAGIC %scala
# MAGIC val total: Long = dbutils.fs.ls("/mnt/mortgage_data/gz/2016").map(f => f.size).sum
# MAGIC println(s"$total bytes")
# MAGIC println(s"${total/1024.0/1024.0} MB")
# MAGIC println(s"${total/1024.0/1024.0/1024.0} GB")

# COMMAND ----------

from pyspark.sql.types import *
'''
fields = [
  StructField('enterprise_flag', StringType(), False),
  StructField('record_number', StringType(), False),
  StructField('usps_code', StringType(), False),
  StructField('msa_code', StringType(), False),
  StructField('county_code', StringType(), False),
  StructField('census_tract', StringType(), False),
  StructField('census_percent_minority', StringType(), False),
  StructField('census_median_income', StringType(), False),
  StructField('local_area_median_income', StringType(), False),
  StructField('tract_income_ratio', StringType(), False),
  StructField('borrowers_annual_income', StringType(), False),
  StructField('area_median_family_income', StringType(), False),
  StructField('borrower_income_ratio', StringType(), False),
  StructField('unpaid_balance', StringType(), False),
  StructField('loan_purpose', StringType(), False),
  StructField('federal_guarantee', StringType(), False),
  StructField('number_of_borrowers', StringType(), False),
  StructField('first_time_buyer', StringType(), False),
  StructField('borrower_race_1', StringType(), False),
  StructField('borrower_race_2', StringType(), False),
  StructField('borrower_race_3', StringType(), False),
  StructField('borrower_race_4', StringType(), False),
  StructField('borrower_race_5', StringType(), False),
  StructField('borrower_ethnicity', StringType(), False),
  StructField('co_borrower_race_1', StringType(), False),
  StructField('co_borrower_race_2', StringType(), False),
  StructField('co_borrower_race_3', StringType(), False),
  StructField('co_borrower_race_4', StringType(), False),
  StructField('co_borrower_race_5', StringType(), False),
  StructField('co_borrower_ethnicity', StringType(), False),
  StructField('borrower_gender', StringType(), False),
  StructField('co_borrower_gender', StringType(), False),
  StructField('age_borrower', StringType(), False),
  StructField('age_co_borrower', StringType(), False),
  StructField('occupancy_code', StringType(), False),
  StructField('rate_spread', StringType(), False),
  StructField('hoepa_status', StringType(), False),
  StructField('property_type', StringType(), False),
  StructField('lien_status', StringType(), False),
  StructField('year', StringType(), False)
]

schema = StructType(fields)
    '''

# COMMAND ----------

from pyspark.sql import functions as f
year = [2010, 2011, 2012, 2013, 2014, 2015, 2016]

for yr in year:
  src = "/mnt/tcruz/mortgage_data/gz/%s/" % (yr)
  dest = "/mnt/tcruz/mortgage_data/parquet/%s/" % (yr)
  print(src)
  print(dest)
  df = spark.read \
    .option("header", "false") \
    .option("inferSchema", "true") \
    .csv(src) \
    .withColumn('year', f.lit(yr)) \
    .toDF(
      'enterprise_flag',
      'record_number',
      'usps_code',
      'msa_code',
      'county_code',
      'census_tract',
      'census_percent_minority',
      'census_median_income',
      'local_area_median_income',
      'tract_income_ratio',
      'borrowers_annual_income',
      'area_median_family_income',
      'borrower_income_ratio',
      'unpaid_balance',
      'loan_purpose',
      'federal_guarantee',
      'number_of_borrowers',
      'first_time_buyer',
      'borrower_race_1',
      'borrower_race_2',
      'borrower_race_3',
      'borrower_race_4',
      'borrower_race_5',
      'borrower_ethnicity',
      'co_borrower_race_1',
      'co_borrower_race_2',
      'co_borrower_race_3',
      'co_borrower_race_4',
      'co_borrower_race_5',
      'co_borrower_ethnicity',
      'borrower_gender',
      'co_borrower_gender',
      'age_borrower',
      'age_co_borrower',
      'occupancy_code',
      'rate_spread',
      'hoepa_status',
      'property_type',
      'lien_status',
      'year')
    
  df.coalesce(1) \
  .write \
  .mode("overwrite") \
  .partitionBy('year') \
  .parquet(dest)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %sh ls -lRh /dbfs/mnt/tcruz/mortgage_data/parquet/

# COMMAND ----------

total = sum(map(lambda f: f.size, dbutils.fs.ls("/mnt/tcruz/mortgage_data/parquet/2016/year=2016")))
print "%s bytes" % total
print "%.2f MB" % (float(total)/1024.0/1024.0)
print "%.2f GB" % (float(total)/1024.0/1024.0/1024.0)

# COMMAND ----------

# MAGIC %scala
# MAGIC val total: Long = dbutils.fs.ls("/mnt/tcruz/mortgage_data/parquet/2016/year=2016").map(f => f.size).sum
# MAGIC println(s"$total bytes")
# MAGIC println(s"${total/1024.0/1024.0} MB")
# MAGIC println(s"${total/1024.0/1024.0/1024.0} GB")

# COMMAND ----------

df = spark.read.parquet("/mnt/mortgage_data/parquet/2010/")
year = [2011, 2012, 2013, 2014, 2015, 2016]
for yr in year:
  src = "/mnt/tcruz/mortgage_data/parquet/%s/" % (yr)
  df = df.union(spark.read.parquet(src))

# COMMAND ----------

print df.count()
print df.rdd.getNumPartitions()
df = df.repartition(4)
print df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %sql 
# MAGIC USE tcruz;
# MAGIC DROP TABLE IF EXISTS mortgage;
# MAGIC CREATE EXTERNAL TABLE mortgage
# MAGIC LOCATION '/mnt/tcruz/mortgage_data/table/'
# MAGIC -- PARTITIONED BY (year) -- can't use this with AS SELECT
# MAGIC -- LOCATION 's3a://databricks-tcruz/mortgage/
# MAGIC AS SELECT * from df

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- no partition information
# MAGIC describe extended tcruz.mortgage

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ALTER TABLE mortgage ADD PARTITION (year)

# COMMAND ----------

# MAGIC %fs ls /mnt/tcruz/mortgage_data/table/

# COMMAND ----------

sql("show create table mortgage").toPandas().to_csv("/dbfs/tmp/create_table")

# COMMAND ----------

# MAGIC %sh cat /dbfs/tmp/create_table

# COMMAND ----------

df.write \
  .mode("overwrite") \
  .partitionBy('year') \
  .parquet('/mnt/tcruz/mortgage_data/table-partitioned')

# COMMAND ----------

# MAGIC %sql
# MAGIC USE tcruz;
# MAGIC DROP TABLE IF EXISTS `mortgage`;
# MAGIC CREATE TABLE `mortgage` (`enterprise_flag` INT, `record_number` INT, `usps_code` INT, `msa_code` INT, `county_code` INT, `census_tract` INT, `census_percent_minority` DOUBLE, `census_median_income` INT, `local_area_median_income` INT, `tract_income_ratio` DOUBLE, `borrowers_annual_income` DECIMAL(10,0), `area_median_family_income` INT, `borrower_income_ratio` DOUBLE, `unpaid_balance` INT, `loan_purpose` INT, `federal_guarantee` INT, `number_of_borrowers` INT, `first_time_buyer` INT, `borrower_race_1` INT, `borrower_race_2` INT, `borrower_race_3` INT, `borrower_race_4` INT, `borrower_race_5` INT, `borrower_ethnicity` INT, `co_borrower_race_1` INT, `co_borrower_race_2` INT, `co_borrower_race_3` INT, `co_borrower_race_4` INT, `co_borrower_race_5` INT, `co_borrower_ethnicity` INT, `borrower_gender` INT, `co_borrower_gender` INT, `age_borrower` INT, `age_co_borrower` INT, `occupancy_code` INT, `rate_spread` DOUBLE, `hoepa_status` INT, `property_type` INT, `lien_status` INT, `year` INT)
# MAGIC USING parquet
# MAGIC 
# MAGIC PARTITIONED BY (year)
# MAGIC 
# MAGIC OPTIONS (
# MAGIC   `serialization.format` '1',
# MAGIC   path 'dbfs:/mnt/tcruz/mortgage_data/table-partitioned'
# MAGIC );
# MAGIC 
# MAGIC -- necessary to create a partitioned table from existing data
# MAGIC -- https://docs.databricks.com/user-guide/tables.html#create-a-partitioned-table
# MAGIC MSCK REPAIR TABLE mortgage;

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- notice partition information
# MAGIC describe extended mortgage

# COMMAND ----------

# MAGIC %fs ls /mnt/tcruz/mortgage_data/table-partitioned/year=2010

# COMMAND ----------

# MAGIC %sql 
# MAGIC use tcruz; 
# MAGIC select count(1) from mortgage

# COMMAND ----------

