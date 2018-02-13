# Databricks notebook source
# MAGIC %md ## Credit Card Fraud Prediction
# MAGIC 
# MAGIC <img src="/files/img/fraud_ml_pipeline.png" alt="workflow" width="500">
# MAGIC 
# MAGIC #### 1. Load raw data
# MAGIC #### 2. Explore data, features
# MAGIC #### 3. Create ML model / Evaluate

# COMMAND ----------

# MAGIC %sql describe extended auth_data

# COMMAND ----------

df = table('auth_data')
df.rdd.getNumPartitions()

# COMMAND ----------

# df.coalesce(8).write.mode('overwrite').parquet('/mnt/tcruz/cc-fraud-p/')

# COMMAND ----------

# MAGIC %sql select count(1) from auth_data

# COMMAND ----------

dataset = df.selectExpr(
  'acct_id_token',
  'auth_id',
  "double(case when FRD_IND = 'Y' then 1.0 else 0.0 end) fraud_reported",
  'cast(AUTHZN_RQST_PROC_DT as bigint) -  cast (ACCT_OPEN_DT as bigint) account_age',
  'cast(PLSTC_ACTVN_DT as bigint) -  cast (ACCT_OPEN_DT as bigint) activation_age',
  'cast(PLSTC_FRST_USE_TS as bigint) -  cast (ACCT_OPEN_DT as bigint) time_since_first_use',
  'int(substring(AUTHZN_RQST_PROC_TM, 0,2)) time_of_day',
  'AUTHZN_AMT',
  'ACCT_AVL_CASH_BEFORE_AMT',
  'ACCT_AVL_MONEY_BEFORE_AMT',
  'ACCT_CL_AMT',
  'ACCT_CURR_BAL',
  'AUTHZN_OUTSTD_AMT',
  'AUTHZN_OUTSTD_CASH_AMT',
  'APPRD_AUTHZN_CNT',
  'APPRD_CASH_AUTHZN_CNT',
  'AUTHZN_CHAR_CD',
  'AUTHZN_CATG_CD',
  'CARD_VFCN_2_RESPNS_CD',
  'CARD_VFCN_2_VLDTN_DUR',
  'POS_ENTRY_MTHD_CD',
  'TRMNL_ATTNDNC_CD',
  'TRMNL_CLASFN_CD',
  'DISTANCE_FROM_HOME' 
)

# COMMAND ----------

display(dataset)

# COMMAND ----------

types = set(map(lambda t: t[1], dataset.dtypes))
print types

# COMMAND ----------

from pyspark.sql.functions import *

strCols = map(lambda t: t[0], filter(lambda t: t[1] == 'string', dataset.dtypes))
intCols = map(lambda t: t[0], filter(lambda t: t[1] == 'int', dataset.dtypes))
bigintCols = map(lambda t: t[0], filter(lambda t: t[1] == 'bigint', dataset.dtypes))
dblCols = map(lambda t: t[0], filter(lambda t: t[1] == 'double', dataset.dtypes))

print 'str: ' + str(strCols)
print 'int: ' + str(intCols)
print 'bigint: ' + str(bigintCols)
print 'dbl: ' + str(dblCols)

# [row_idx][json_idx]
strColsCount = sorted(map(lambda c: (c, dataset.select(countDistinct(c)).collect()[0][0]), strCols), key=lambda x: x[1], reverse=True)
intColsCount = sorted(map(lambda c: (c, dataset.select(countDistinct(c)).collect()[0][0]), intCols), key=lambda x: x[1], reverse=True)
bigintColsCount = sorted(map(lambda c: (c, dataset.select(countDistinct(c)).collect()[0][0]), bigintCols), key=lambda x: x[1], reverse=True)
dblColsCount = sorted(map(lambda c: (c, dataset.select(countDistinct(c)).collect()[0][0]), dblCols), key=lambda x: x[1], reverse=True)

# COMMAND ----------

display(strColsCount)

# COMMAND ----------

display(df.describe('CARD_VFCN_2_VLDTN_DUR'))

# COMMAND ----------

display(df.select(countDistinct('CARD_VFCN_2_VLDTN_DUR')))

# COMMAND ----------

# MAGIC %sql select CARD_VFCN_2_VLDTN_DUR, count(1)
# MAGIC from auth_data
# MAGIC group by 1 order by 2 desc

# COMMAND ----------

display(intColsCount)

# COMMAND ----------

display(bigintColsCount)

# COMMAND ----------

display(dblColsCount)

# COMMAND ----------

authDataSet = dataset.na.fill(0.0).na.fill("N/A")

# COMMAND ----------

display(authDataSet)

# COMMAND ----------

categorical = ['AUTHZN_CHAR_CD', 'CARD_VFCN_2_RESPNS_CD', 'time_of_day', 'APPRD_CASH_AUTHZN_CNT', 'AUTHZN_CATG_CD', 'POS_ENTRY_MTHD_CD', 'TRMNL_CLASFN_CD', 'TRMNL_ATTNDNC_CD']

# CARD_VFCN_2_VLDTN_DUR categorical?
numeric = ['APPRD_AUTHZN_CNT', 'time_since_first_use', 'account_age', 'activation_age','DISTANCE_FROM_HOME', 'ACCT_AVL_MONEY_BEFORE_AMT', 'ACCT_CURR_BAL', 'AUTHZN_OUTSTD_AMT', 'ACCT_AVL_CASH_BEFORE_AMT', 'AUTHZN_AMT', 'AUTHZN_OUTSTD_CASH_AMT', 'ACCT_CL_AMT']

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

# categorical features
stringIndexers = map(lambda c: StringIndexer(inputCol = c, outputCol = c + "_idx"), categorical)
catAssembler = VectorAssembler(inputCols = map(lambda c: c + "_idx", categorical), outputCol = 'catFeatures')

# numeric features
numAssembler = VectorAssembler(inputCols = numeric, outputCol='numericFeatures')
scaler = StandardScaler(inputCol = 'numericFeatures', outputCol = 'scaledFeatures' )

assembler = VectorAssembler(inputCols = ['catFeatures', 'scaledFeatures'], outputCol = 'features')

stages = stringIndexers + [catAssembler] + [numAssembler] + [scaler] + [assembler]

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=stages)

featurizedDataset = pipeline.fit(authDataSet).transform(authDataSet)

# COMMAND ----------

(train, test) = featurizedDataset.randomSplit([0.7, 0.3], seed=42)
train.cache()
test.cache()

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lr = LogisticRegression().setTol(1e-9).setFeaturesCol("features").setLabelCol("fraud_reported").setRegParam(0.01)
model = lr.fit(train)
predictions = model.transform(test)

# COMMAND ----------

ev = BinaryClassificationEvaluator().setLabelCol("fraud_reported").setMetricName("areaUnderROC")
print ev.evaluate(predictions)

ev = BinaryClassificationEvaluator().setLabelCol("fraud_reported").setMetricName("areaUnderPR")
print ev.evaluate(predictions)

# COMMAND ----------

weights = map(lambda w: '%.10f' % w, model.coefficients.toArray())
weightedFeatures = sorted(zip(categorical + numeric, weights), key=lambda x: x[1], reverse=True)
weightedFeatures

# COMMAND ----------

spark.createDataFrame(weightedFeatures).toDF("feature", "weight").createOrReplaceTempView('wf')

# COMMAND ----------

# MAGIC %sql select feature, weight from wf order by weight desc

# COMMAND ----------

display(predictions.select('prediction', 'fraud_reported', 'probability', 'rawPrediction'))

# COMMAND ----------

predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC select sum(case when fraud_reported = prediction then 1 else 0 end) / (count(1) * 1.0) as match_pct
# MAGIC from predictions

# COMMAND ----------

# MAGIC %sql select time_of_day,
# MAGIC sum(case when prediction = fraud_reported then 1 else 0 end) as match,
# MAGIC sum(case when prediction != fraud_reported then 1 else 0 end) as no_match
# MAGIC from predictions
# MAGIC group by 1 order by 1

# COMMAND ----------

# MAGIC %sql select time_of_day,
# MAGIC sum(case when prediction = fraud_reported then 1 else 0 end) / (count(*) * 1.0) as match_pct
# MAGIC from predictions
# MAGIC group by 1 order by 1