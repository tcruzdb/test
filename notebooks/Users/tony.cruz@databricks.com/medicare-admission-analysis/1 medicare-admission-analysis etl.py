# Databricks notebook source
# MAGIC %md ## Hospital Admission Analysis
# MAGIC [**Kaggle Dataset on Github**](https://github.com/jiunjiunma/heritage-health-prize/blob/master/modeling_set1.csv)
# MAGIC 
# MAGIC <img src="/files/img/fraud_ml_pipeline.png" alt="workflow" width="500">
# MAGIC 
# MAGIC #### 1. Load raw data
# MAGIC #### 2. Explore data, features
# MAGIC #### 3. Create ML model / Evaluate

# COMMAND ----------

# MAGIC %sh
# MAGIC if [ ! -d /tmp/patient ]; then mkdir /tmp/patient; fi
# MAGIC wget https://raw.githubusercontent.com/jiunjiunma/heritage-health-prize/master/modeling_set1.csv -O /tmp/patient/modeling_set1.csv
# MAGIC ls -lh /tmp/patient/

# COMMAND ----------

# MAGIC %sh head /tmp/patient/modeling_set1.csv

# COMMAND ----------

# MAGIC %sh wc -l /tmp/patient/modeling_set1.csv

# COMMAND ----------

# MAGIC %sh cat /tmp/patient/modeling_set1.csv | awk -F, '{print NF}' |sort|uniq -c|sort -rn

# COMMAND ----------

dbutils.fs.mkdirs("/mnt/patient")
dbutils.fs.mv("file:///tmp/patient/modeling_set1.csv", "/mnt/patient/modeling_set1.csv")

# COMMAND ----------

# MAGIC %fs ls /mnt/patient

# COMMAND ----------

# MAGIC %fs head /mnt/patient/modeling_set1.csv

# COMMAND ----------

rawDf = spark.read \
.option("header", "true") \
.option("inferSchema", "true") \
.csv("dbfs:/mnt/patient/modeling_set1.csv")

# COMMAND ----------

rawDf.coalesce(1).write.mode('overwrite').format('parquet').saveAsTable("tc_patient_admissions")

# COMMAND ----------

# MAGIC %sql describe extended tc_patient_admissions

# COMMAND ----------

# MAGIC %sql select count(*) from tc_patient_admissions

# COMMAND ----------

# MAGIC %sql select distinct age_MISS from tc_patient_admissions limit 10

# COMMAND ----------

# MAGIC %sql select daysinhospital, 
# MAGIC   sum(case when age_05 = 1 then 1 else 0 end) as age_05,
# MAGIC   sum(case when age_15 = 1 then 1 else 0 end) as age_15,
# MAGIC   sum(case when age_25 = 1 then 1 else 0 end) as age_25,
# MAGIC   sum(case when age_35 = 1 then 1 else 0 end) as age_35,
# MAGIC   sum(case when age_45 = 1 then 1 else 0 end) as age_45,
# MAGIC   sum(case when age_55 = 1 then 1 else 0 end) as age_55,
# MAGIC   sum(case when age_65 = 1 then 1 else 0 end) as age_65,
# MAGIC   sum(case when age_75 = 1 then 1 else 0 end) as age_75,
# MAGIC   sum(case when age_85 = 1 then 1 else 0 end) as age_85,
# MAGIC   sum(case when age_miss = 1 then 1 else 0 end) as age_miss
# MAGIC from tc_patient_admissions
# MAGIC where daysinhospital is not null and daysinhospital > 0
# MAGIC group by 1
# MAGIC order by 1

# COMMAND ----------

# MAGIC %sql select max(daysinhospital) from tc_patient_admissions

# COMMAND ----------

# MAGIC %sql select daysinhospital, 
# MAGIC   sum(case when age_05 = 1 then 1 else 0 end) as age_05,
# MAGIC   sum(case when age_15 = 1 then 1 else 0 end) as age_15,
# MAGIC   sum(case when age_25 = 1 then 1 else 0 end) as age_25,
# MAGIC   sum(case when age_35 = 1 then 1 else 0 end) as age_35,
# MAGIC   sum(case when age_45 = 1 then 1 else 0 end) as age_45,
# MAGIC   sum(case when age_55 = 1 then 1 else 0 end) as age_55,
# MAGIC   sum(case when age_65 = 1 then 1 else 0 end) as age_65,
# MAGIC   sum(case when age_75 = 1 then 1 else 0 end) as age_75,
# MAGIC   sum(case when age_85 = 1 then 1 else 0 end) as age_85,
# MAGIC   sum(case when age_miss = 1 then 1 else 0 end) as age_miss
# MAGIC from tc_patient_admissions
# MAGIC where daysinhospital is not null and daysinhospital > 10
# MAGIC group by 1
# MAGIC order by 1

# COMMAND ----------

# MAGIC %sql select
# MAGIC count(distinct MemberID_t) as MemberID_t,
# MAGIC count(distinct YEAR_t) as YEAR_t,
# MAGIC count(distinct ClaimsTruncated) as ClaimsTruncated,
# MAGIC count(distinct DaysInHospital) as DaysInHospital,
# MAGIC count(distinct trainset) as trainset,
# MAGIC count(distinct age_05) as age_05,
# MAGIC count(distinct age_15) as age_15,
# MAGIC count(distinct age_25) as age_25,
# MAGIC count(distinct age_35) as age_35,
# MAGIC count(distinct age_45) as age_45,
# MAGIC count(distinct age_55) as age_55,
# MAGIC count(distinct age_65) as age_65,
# MAGIC count(distinct age_75) as age_75,
# MAGIC count(distinct age_85) as age_85,
# MAGIC count(distinct age_MISS) as age_MISS,
# MAGIC count(distinct sexMALE) as sexMALE,
# MAGIC count(distinct sexFEMALE) as sexFEMALE,
# MAGIC count(distinct sexMISS) as sexMISS,
# MAGIC count(distinct no_Claims) as no_Claims,
# MAGIC count(distinct no_Providers) as no_Providers,
# MAGIC count(distinct no_Vendors) as no_Vendors,
# MAGIC count(distinct no_PCPs) as no_PCPs,
# MAGIC count(distinct no_PlaceSvcs) as no_PlaceSvcs,
# MAGIC count(distinct no_Specialities) as no_Specialities,
# MAGIC count(distinct no_PrimaryConditionGroups) as no_PrimaryConditionGroups,
# MAGIC count(distinct no_ProcedureGroups) as no_ProcedureGroups,
# MAGIC count(distinct PayDelay_max) as PayDelay_max,
# MAGIC count(distinct PayDelay_min) as PayDelay_min,
# MAGIC count(distinct PayDelay_ave) as PayDelay_ave,
# MAGIC count(distinct PayDelay_stdev) as PayDelay_stdev,
# MAGIC count(distinct LOS_max) as LOS_max,
# MAGIC count(distinct LOS_min) as LOS_min,
# MAGIC count(distinct LOS_ave) as LOS_ave,
# MAGIC count(distinct LOS_stdev) as LOS_stdev,
# MAGIC count(distinct LOS_TOT_UNKNOWN) as LOS_TOT_UNKNOWN,
# MAGIC count(distinct LOS_TOT_SUPRESSED) as LOS_TOT_SUPRESSED,
# MAGIC count(distinct LOS_TOT_KNOWN) as LOS_TOT_KNOWN,
# MAGIC count(distinct dsfs_max) as dsfs_max,
# MAGIC count(distinct dsfs_min) as dsfs_min,
# MAGIC count(distinct dsfs_range) as dsfs_range,
# MAGIC count(distinct dsfs_ave) as dsfs_ave,
# MAGIC count(distinct dsfs_stdev) as dsfs_stdev,
# MAGIC count(distinct CharlsonIndexI_max) as CharlsonIndexI_max,
# MAGIC count(distinct CharlsonIndexI_min) as CharlsonIndexI_min,
# MAGIC count(distinct CharlsonIndexI_ave) as CharlsonIndexI_ave,
# MAGIC count(distinct CharlsonIndexI_range) as CharlsonIndexI_range,
# MAGIC count(distinct CharlsonIndexI_stdev) as CharlsonIndexI_stdev,
# MAGIC count(distinct pcg1) as pcg1,
# MAGIC count(distinct pcg2) as pcg2,
# MAGIC count(distinct pcg3) as pcg3,
# MAGIC count(distinct pcg4) as pcg4,
# MAGIC count(distinct pcg5) as pcg5,
# MAGIC count(distinct pcg6) as pcg6,
# MAGIC count(distinct pcg7) as pcg7,
# MAGIC count(distinct pcg8) as pcg8,
# MAGIC count(distinct pcg9) as pcg9,
# MAGIC count(distinct pcg10) as pcg10,
# MAGIC count(distinct pcg11) as pcg11,
# MAGIC count(distinct pcg12) as pcg12,
# MAGIC count(distinct pcg13) as pcg13,
# MAGIC count(distinct pcg14) as pcg14,
# MAGIC count(distinct pcg15) as pcg15,
# MAGIC count(distinct pcg16) as pcg16,
# MAGIC count(distinct pcg17) as pcg17,
# MAGIC count(distinct pcg18) as pcg18,
# MAGIC count(distinct pcg19) as pcg19,
# MAGIC count(distinct pcg20) as pcg20,
# MAGIC count(distinct pcg21) as pcg21,
# MAGIC count(distinct pcg22) as pcg22,
# MAGIC count(distinct pcg23) as pcg23,
# MAGIC count(distinct pcg24) as pcg24,
# MAGIC count(distinct pcg25) as pcg25,
# MAGIC count(distinct pcg26) as pcg26,
# MAGIC count(distinct pcg27) as pcg27,
# MAGIC count(distinct pcg28) as pcg28,
# MAGIC count(distinct pcg29) as pcg29,
# MAGIC count(distinct pcg30) as pcg30,
# MAGIC count(distinct pcg31) as pcg31,
# MAGIC count(distinct pcg32) as pcg32,
# MAGIC count(distinct pcg33) as pcg33,
# MAGIC count(distinct pcg34) as pcg34,
# MAGIC count(distinct pcg35) as pcg35,
# MAGIC count(distinct pcg36) as pcg36,
# MAGIC count(distinct pcg37) as pcg37,
# MAGIC count(distinct pcg38) as pcg38,
# MAGIC count(distinct pcg39) as pcg39,
# MAGIC count(distinct pcg40) as pcg40,
# MAGIC count(distinct pcg41) as pcg41,
# MAGIC count(distinct pcg42) as pcg42,
# MAGIC count(distinct pcg43) as pcg43,
# MAGIC count(distinct pcg44) as pcg44,
# MAGIC count(distinct pcg45) as pcg45,
# MAGIC count(distinct pcg46) as pcg46,
# MAGIC count(distinct sp1) as sp1,
# MAGIC count(distinct sp2) as sp2,
# MAGIC count(distinct sp3) as sp3,
# MAGIC count(distinct sp4) as sp4,
# MAGIC count(distinct sp5) as sp5,
# MAGIC count(distinct sp6) as sp6,
# MAGIC count(distinct sp7) as sp7,
# MAGIC count(distinct sp8) as sp8,
# MAGIC count(distinct sp9) as sp9,
# MAGIC count(distinct sp10) as sp10,
# MAGIC count(distinct sp11) as sp11,
# MAGIC count(distinct sp12) as sp12,
# MAGIC count(distinct sp13) as sp13,
# MAGIC count(distinct pg1) as pg1,
# MAGIC count(distinct pg2) as pg2,
# MAGIC count(distinct pg3) as pg3,
# MAGIC count(distinct pg4) as pg4,
# MAGIC count(distinct pg5) as pg5,
# MAGIC count(distinct pg6) as pg6,
# MAGIC count(distinct pg7) as pg7,
# MAGIC count(distinct pg8) as pg8,
# MAGIC count(distinct pg9) as pg9,
# MAGIC count(distinct pg10) as pg10,
# MAGIC count(distinct pg11) as pg11,
# MAGIC count(distinct pg12) as pg12,
# MAGIC count(distinct pg13) as pg13,
# MAGIC count(distinct pg14) as pg14,
# MAGIC count(distinct pg15) as pg15,
# MAGIC count(distinct pg16) as pg16,
# MAGIC count(distinct pg17) as pg17,
# MAGIC count(distinct pg18) as pg18,
# MAGIC count(distinct ps1) as ps1,
# MAGIC count(distinct ps2) as ps2,
# MAGIC count(distinct ps3) as ps3,
# MAGIC count(distinct ps4) as ps4,
# MAGIC count(distinct ps5) as ps5,
# MAGIC count(distinct ps6) as ps6,
# MAGIC count(distinct ps7) as ps7,
# MAGIC count(distinct ps8) as ps8,
# MAGIC count(distinct ps9) as ps9,
# MAGIC count(distinct drugCount_max) as drugCount_max,
# MAGIC count(distinct drugCount_min) as drugCount_min,
# MAGIC count(distinct drugCount_ave) as drugCount_ave,
# MAGIC count(distinct drugcount_months) as drugcount_months,
# MAGIC count(distinct labCount_max) as labCount_max,
# MAGIC count(distinct labCount_min) as labCount_min,
# MAGIC count(distinct labCount_ave) as labCount_ave,
# MAGIC count(distinct labcount_months) as labcount_months,
# MAGIC count(distinct labNull) as labNull,
# MAGIC count(distinct drugNull) as drugNull
# MAGIC from tc_patient_admissions