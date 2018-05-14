# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://assets.cms.gov/resources/cms/images/logo/site-logo.png" />
# MAGIC 
# MAGIC ### [CMS Dataset Downloads](https://www.cms.gov/OpenPayments/Explore-the-Data/Dataset-Downloads.html): General Payments 2016
# MAGIC 
# MAGIC * `OP_DTL_GNRL_PGYR2016_P01172018.csv: 
# MAGIC This file contains the data set of General Payments reported for the 2016 program year.`

# COMMAND ----------

# MAGIC %md #### ETL raw general payments csv data from CMS.gov to a parquet-backed table in the data catalog/metastore.  Use Spark for schema inference.

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

cms2016.createOrReplaceTempView("cms2016")

# COMMAND ----------

# MAGIC %sql 
# MAGIC create database if not exists cms; 
# MAGIC use cms

# COMMAND ----------

# MAGIC %sql drop table if exists cms.gp2016

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE EXTERNAL TABLE IF NOT EXISTS cms.gp2016
# MAGIC -- USING PARQUET
# MAGIC -- 'location' keyword implies 'external' table
# MAGIC LOCATION '/mnt/tcruz/hls/cms/gp/2016/parquet/'
# MAGIC -- notice we didn't specify schema in 'create table' above
# MAGIC AS SELECT * FROM cms2016
# MAGIC -- this will write out the dataframe as parquet files to specified location

# COMMAND ----------

# MAGIC %sql describe extended cms.gp2016

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Now that we have a table in the metastore with the inferred schema, we can use the `CREATE TABLE` statement from that table and point it at existing parquet files with the *same* schema, e.g. `/mnt/tcruz/hls/cms/gp/2016/parquet-existing`

# COMMAND ----------

# MAGIC %sh ls -lha /dbfs/mnt/tcruz/hls/cms/gp/2016/parquet-existing/

# COMMAND ----------

sql("show create table cms.gp2016").toPandas().to_csv('/dbfs/mnt/tcruz/create_table')

# COMMAND ----------

# MAGIC %sh cat /dbfs/mnt/tcruz/create_table

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists `cms`.`gp2016existing`;
# MAGIC CREATE TABLE `cms`.`gp2016existing` (`Change_Type` STRING, `Covered_Recipient_Type` STRING, `Teaching_Hospital_CCN` INT, `Teaching_Hospital_ID` INT, `Teaching_Hospital_Name` STRING, `Physician_Profile_ID` INT, `Physician_First_Name` STRING, `Physician_Middle_Name` STRING, `Physician_Last_Name` STRING, `Physician_Name_Suffix` STRING, `Recipient_Primary_Business_Street_Address_Line1` STRING, `Recipient_Primary_Business_Street_Address_Line2` STRING, `Recipient_City` STRING, `Recipient_State` STRING, `Recipient_Zip_Code` STRING, `Recipient_Country` STRING, `Recipient_Province` STRING, `Recipient_Postal_Code` STRING, `Physician_Primary_Type` STRING, `Physician_Specialty` STRING, `Physician_License_State_code1` STRING, `Physician_License_State_code2` STRING, `Physician_License_State_code3` STRING, `Physician_License_State_code4` STRING, `Physician_License_State_code5` STRING, `Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name` STRING, `Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID` STRING, `Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name` STRING, `Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_State` STRING, `Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country` STRING, `Total_Amount_of_Payment_USDollars` STRING, `Date_of_Payment` STRING, `Number_of_Payments_Included_in_Total_Amount` STRING, `Form_of_Payment_or_Transfer_of_Value` STRING, `Nature_of_Payment_or_Transfer_of_Value` STRING, `City_of_Travel` STRING, `State_of_Travel` STRING, `Country_of_Travel` STRING, `Physician_Ownership_Indicator` STRING, `Third_Party_Payment_Recipient_Indicator` STRING, `Name_of_Third_Party_Entity_Receiving_Payment_or_Transfer_of_Value` STRING, `Charity_Indicator` STRING, `Third_Party_Equals_Covered_Recipient_Indicator` STRING, `Contextual_Information` STRING, `Delay_in_Publication_Indicator` STRING, `Record_ID` STRING, `Dispute_Status_for_Publication` STRING, `Related_Product_Indicator` STRING, `Covered_or_Noncovered_Indicator_1` STRING, `Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1` STRING, `Product_Category_or_Therapeutic_Area_1` STRING, `Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1` STRING, `Associated_Drug_or_Biological_NDC_1` STRING, `Covered_or_Noncovered_Indicator_2` STRING, `Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_2` STRING, `Product_Category_or_Therapeutic_Area_2` STRING, `Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_2` STRING, `Associated_Drug_or_Biological_NDC_2` STRING, `Covered_or_Noncovered_Indicator_3` STRING, `Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_3` STRING, `Product_Category_or_Therapeutic_Area_3` STRING, `Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_3` STRING, `Associated_Drug_or_Biological_NDC_3` STRING, `Covered_or_Noncovered_Indicator_4` STRING, `Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_4` STRING, `Product_Category_or_Therapeutic_Area_4` STRING, `Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_4` STRING, `Associated_Drug_or_Biological_NDC_4` STRING, `Covered_or_Noncovered_Indicator_5` STRING, `Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_5` STRING, `Product_Category_or_Therapeutic_Area_5` STRING, `Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_5` STRING, `Associated_Drug_or_Biological_NDC_5` STRING, `Program_Year` INT, `Payment_Publication_Date` STRING)
# MAGIC USING parquet
# MAGIC OPTIONS (
# MAGIC   `serialization.format` '1',
# MAGIC   
# MAGIC   -- updated path to existing parquet files
# MAGIC   path 'dbfs:/mnt/tcruz/hls/cms/gp/2016/parquet-existing'
# MAGIC )

# COMMAND ----------

# MAGIC %sql describe extended cms.gp2016existing

# COMMAND ----------

# MAGIC %sql use cms;
# MAGIC show tables;

# COMMAND ----------

# MAGIC %sql select count(*) from cms.gp2016existing

# COMMAND ----------

# MAGIC %sql
# MAGIC use cms;
# MAGIC 
# MAGIC select length(recipient_state), count(1)
# MAGIC from gp2016existing
# MAGIC group by 1 
# MAGIC order by 2 desc

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

''' just an example of syntax to create dataframe from JDBC datasource

states = spark.read \
  .jdbc("jdbc:mysql://host.server.com:3306?user=user&password=password", "database.properstates") \
  .select('state_abbr') \
  .distinct() \
  .orderBy('state_abbr') 
  '''

# COMMAND ----------

states = spark.read.option('header', False).csv('/FileStore/state_codes.csv').toDF("state_abbr")

# COMMAND ----------

display(states)

# COMMAND ----------

cmsgeo = cms2016.join(states, states.state_abbr == cms2016.Recipient_State)

# COMMAND ----------

print(cms2016.count())
print(cmsgeo.count())
print(cms2016.count() - cmsgeo.count())

# COMMAND ----------

cmsgeo.createOrReplaceTempView("cmsgeo")

# COMMAND ----------

# MAGIC %sql drop table if exists cms.gp2016geo

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE EXTERNAL TABLE IF NOT EXISTS cms.gp2016geo
# MAGIC -- using parquet
# MAGIC -- 'location' keyword implies 'external' table
# MAGIC LOCATION '/mnt/tcruz/hls/cms/gpgeo/2016/parquet/'
# MAGIC -- notice we didn't specify schema in 'create table' above
# MAGIC AS SELECT * FROM cmsgeo
# MAGIC -- this will write out the dataframe as parquet files to specified location

# COMMAND ----------

# MAGIC %fs ls /mnt/tcruz/hls/cms/gpgeo/2016/parquet/

# COMMAND ----------

# MAGIC %sql describe extended cms.gp2016geo

# COMMAND ----------

# MAGIC %sql
# MAGIC select recipient_state, count(1) as count
# MAGIC from cms.gp2016geo
# MAGIC where length(recipient_state) = 2
# MAGIC group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select recipient_state, count(1) as count
# MAGIC from cms.gp2016geo
# MAGIC where length(recipient_state) = 2
# MAGIC group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %sql 
# MAGIC select recipient_state, sum(cast(total_amount_of_payment_usdollars as double)) as total
# MAGIC from cms.gp2016geo
# MAGIC where total_amount_of_payment_usdollars is not null
# MAGIC group by 1 order by 2 desc

# COMMAND ----------

# MAGIC %sql 
# MAGIC select recipient_state, sum(cast(total_amount_of_payment_usdollars as double)) as total
# MAGIC from cms.gp2016geo
# MAGIC where total_amount_of_payment_usdollars is not null
# MAGIC group by 1 order by 2 desc

# COMMAND ----------

