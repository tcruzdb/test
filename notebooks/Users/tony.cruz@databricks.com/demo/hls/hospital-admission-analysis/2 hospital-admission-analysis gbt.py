# Databricks notebook source
# MAGIC %md ## Hospital Admission Analysis
# MAGIC [**Kaggle Dataset on Github**](https://github.com/jiunjiunma/heritage-health-prize/blob/master/modeling_set1.csv)
# MAGIC 
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/pub-tc/ML-workflow.png" width="800">
# MAGIC #### 1. Load raw data
# MAGIC #### 2. Explore data, features
# MAGIC #### 3. Create ML model / Evaluate

# COMMAND ----------

rawDf = table('tc_patient_admissions')

# COMMAND ----------

from pyspark.sql.types import *

# trainset has 1 distinct value after removing rows with null values
df = rawDf.dropna() \
  .drop('MemberID_t') \
  .drop('trainset')

# add new column DaysInHospitalDbl for Binarizer
df = df.selectExpr("*", "cast(DaysInHospital as double) DaysInHospitalDbl").repartition(128)

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

df.columns

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

# MAGIC %sql
# MAGIC select daysinhospital,
# MAGIC sum(case when age_85 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_85,
# MAGIC sum(case when age_75 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_75,
# MAGIC sum(case when age_65 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_65,
# MAGIC sum(case when age_55 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_55,
# MAGIC sum(case when age_45 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_45,
# MAGIC sum(case when age_35 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_35,
# MAGIC sum(case when age_25 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_25,
# MAGIC sum(case when age_15 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_15,
# MAGIC sum(case when age_05 = 1 then 1 else 0 end) / (count(1) * 1.0) as age_05
# MAGIC from tc_patient_admissions
# MAGIC where daysinhospital is not null and daysinhospital > 0
# MAGIC group by 1
# MAGIC order by 1

# COMMAND ----------

from pyspark.ml.feature import Binarizer, StringIndexer, OneHotEncoder, VectorAssembler
import re

# separate categorical and numerical features
patterns = ["^.+_stdev", "^.+_max", "^.+_min", "^.+_ave"]
lists = map(lambda p: filter(re.compile(p).search, df.columns), patterns)
numerical = [col for sublist in lists for col in sublist]

categorical = filter(lambda c: c not in numerical, df.columns)
# what we are trying to predict binarized: admit or not
categorical.remove('DaysInHospital')
categorical.remove('DaysInHospitalDbl')

binarizer = Binarizer()  \
  .setInputCol("DaysInHospitalDbl") \
  .setOutputCol("_admitted") \
  .setThreshold(0.0)

stringIndexers = map(lambda c: StringIndexer(inputCol = c, outputCol = c + "_idx"), categorical)
oneHotEncoders = map(lambda c: OneHotEncoder(inputCol = c + "_idx", outputCol = c + "_idx_ohe"), categorical)

assemblerInputs = map(lambda c: c + "_idx_ohe", categorical)
assemblerInputs += numerical
vectorAssembler = VectorAssembler(inputCols = assemblerInputs, outputCol = "features")

labelStringIndexer = StringIndexer(inputCol = "_admitted", outputCol = "label")

stages = [binarizer] + stringIndexers + oneHotEncoders + [vectorAssembler] + [labelStringIndexer]

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages = stages)

featurizer = pipeline.fit(df)

# COMMAND ----------

display(df)

# COMMAND ----------

featurizedDf = featurizer.transform(df)

# COMMAND ----------

display(featurizedDf.select("features", "label"))

# COMMAND ----------

train, test = featurizedDf.select(["label", "features"]).randomSplit([0.7, 0.3], 42)
train.cache()
test.cache()

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

gbtClassifier = GBTClassifier(labelCol="label", featuresCol="features", maxBins=50, maxDepth=10, maxIter=10)

gbtModel = gbtClassifier.fit(train)

# COMMAND ----------

predictions = gbtModel.transform(test)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
ev = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", metricName="areaUnderROC")
print ev.evaluate(predictions)

# COMMAND ----------

predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql select
# MAGIC prediction,
# MAGIC sum(case when prediction = label then 1 else 0 end) as match,
# MAGIC sum(case when prediction != label then 1 else 0 end) as no_match 
# MAGIC from predictions
# MAGIC group by 1
# MAGIC order by 1

# COMMAND ----------

# MAGIC %sql select
# MAGIC sum(case when prediction = label then 1 else 0 end) / (count(1) * 1.0) as accuracy
# MAGIC from predictions

# COMMAND ----------

