# Databricks notebook source
# MAGIC %md ## Credit Card Fraud Prediction
# MAGIC 
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/pub-tc/ML-workflow.png" width="800">
# MAGIC 
# MAGIC #### 1. Load data
# MAGIC #### 2. Explore data, features
# MAGIC #### 3. Create ML model / Evaluate

# COMMAND ----------

# MAGIC %sql describe extended auth_data

# COMMAND ----------

# MAGIC %sql select count(1) from auth_data

# COMMAND ----------

dataset = table('auth_data').selectExpr(
  'double(case when FRD_IND = "Y" then 1.0 else 0.0 end) FRAUD_REPORTED',
  
  'cast(cast(AUTHZN_RQST_PROC_DT as bigint) - cast (ACCT_OPEN_DT as bigint) as double) ACCOUNT_AGE',
  'cast(cast(PLSTC_ACTVN_DT as bigint) - cast (ACCT_OPEN_DT as bigint) as double) ACTIVATION_AGE',
  'cast(cast(PLSTC_FRST_USE_TS as bigint) - cast (ACCT_OPEN_DT as bigint) as double) TIME_UNTIL_FIRST_USE',
  
  'int(substring(AUTHZN_RQST_PROC_TM, 0,2)) HOUR_OF_DAY',
  
  'AUTHZN_AMT',
  'ACCT_AVL_CASH_BEFORE_AMT',
  'ACCT_AVL_MONEY_BEFORE_AMT',
  'ACCT_CL_AMT',
  'ACCT_CURR_BAL',
  'AUTHZN_OUTSTD_AMT',
  'AUTHZN_OUTSTD_CASH_AMT',
  'APPRD_AUTHZN_CNT',
  'APPRD_CASH_AUTHZN_CNT',
  
  # 'AUTHZN_CHAR_CD',
  # 'AUTHZN_CATG_CD',
  # 'CARD_VFCN_2_RESPNS_CD',
  
  'CARD_VFCN_2_VLDTN_DUR',
  'POS_ENTRY_MTHD_CD',
  
  # 'TRMNL_ATTNDNC_CD',
  
  'TRMNL_CLASFN_CD',
  'DISTANCE_FROM_HOME'
)

# COMMAND ----------

display(dataset)

# COMMAND ----------

dataset.createOrReplaceTempView("df")

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC count(1),
# MAGIC count(distinct(FRAUD_REPORTED)) as FRAUD_REPORTED,
# MAGIC count(distinct(ACCOUNT_AGE)) as ACCOUNT_AGE,
# MAGIC count(distinct(ACTIVATION_AGE)) as ACTIVATION_AGE,
# MAGIC count(distinct(TIME_UNTIL_FIRST_USE)) as TIME_UNTIL_FIRST_USE,
# MAGIC count(distinct(HOUR_OF_DAY)) as HOUR_OF_DAY,
# MAGIC count(distinct(AUTHZN_AMT)) as AUTHZN_AMT,
# MAGIC count(distinct(ACCT_AVL_CASH_BEFORE_AMT)) as ACCT_AVL_CASH_BEFORE_AMT,
# MAGIC count(distinct(ACCT_AVL_MONEY_BEFORE_AMT)) as ACCT_AVL_MONEY_BEFORE_AMT,
# MAGIC count(distinct(ACCT_CL_AMT)) as ACCT_CL_AMT,
# MAGIC count(distinct(ACCT_CURR_BAL)) as ACCT_CURR_BAL,
# MAGIC count(distinct(AUTHZN_OUTSTD_AMT)) as AUTHZN_OUTSTD_AMT,
# MAGIC count(distinct(AUTHZN_OUTSTD_CASH_AMT)) as AUTHZN_OUTSTD_CASH_AMT,
# MAGIC count(distinct(APPRD_AUTHZN_CNT)) as APPRD_AUTHZN_CNT,
# MAGIC count(distinct(APPRD_CASH_AUTHZN_CNT)) as APPRD_CASH_AUTHZN_CNT,
# MAGIC -- count(distinct(AUTHZN_CHAR_CD)) as AUTHZN_CHAR_CD,
# MAGIC -- count(distinct(AUTHZN_CATG_CD)) as AUTHZN_CATG_CD,
# MAGIC -- count(distinct(CARD_VFCN_2_RESPNS_CD)) as CARD_VFCN_2_RESPNS_CD,
# MAGIC count(distinct(CARD_VFCN_2_VLDTN_DUR)) as CARD_VFCN_2_VLDTN_DUR,
# MAGIC count(distinct(POS_ENTRY_MTHD_CD)) as POS_ENTRY_MTHD_CD,
# MAGIC -- count(distinct(TRMNL_ATTNDNC_CD)) as TRMNL_ATTNDNC_CD,
# MAGIC count(distinct(TRMNL_CLASFN_CD)) as TRMNL_CLASFN_CD,
# MAGIC count(distinct(DISTANCE_FROM_HOME)) as DISTANCE_FROM_HOME
# MAGIC from df

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC count(1),
# MAGIC -- count(distinct(AUTHZN_CHAR_CD)) as AUTHZN_CHAR_CD,
# MAGIC -- count(distinct(AUTHZN_CATG_CD)) as AUTHZN_CATG_CD,
# MAGIC -- count(distinct(CARD_VFCN_2_RESPNS_CD)) as CARD_VFCN_2_RESPNS_CD,
# MAGIC count(distinct(POS_ENTRY_MTHD_CD)) as POS_ENTRY_MTHD_CD,
# MAGIC -- count(distinct(TRMNL_ATTNDNC_CD)) as TRMNL_ATTNDNC_CD,
# MAGIC count(distinct(TRMNL_CLASFN_CD)) as TRMNL_CLASFN_CD
# MAGIC from df

# COMMAND ----------

dataset.dtypes

# COMMAND ----------

from pyspark.sql.functions import *

intCols = list(map(lambda t: t[0], filter(lambda t: t[1] == 'int', dataset.dtypes)))
dblCols = list(map(lambda t: t[0], filter(lambda t: t[1] == 'double', dataset.dtypes)))

print(intCols)
print(dblCols)

# COMMAND ----------

# dataset.select(countDistinct(c)).collect() returns list of Row
# [row_idx][json_idx]
intColsCount = sorted(map(lambda c: (c, dataset.select(countDistinct(c)).collect()[0][0]), intCols), key=lambda kv: kv[1], reverse=True)
dblColsCount = sorted(map(lambda c: (c, dataset.select(countDistinct(c)).collect()[0][0]), dblCols), key=lambda kv: kv[1], reverse=True)

# COMMAND ----------

display(intColsCount)

# COMMAND ----------

display(dblColsCount)

# COMMAND ----------

authDataSet = dataset.withColumnRenamed('FRAUD_REPORTED', 'label')
dblCols.remove('FRAUD_REPORTED')

# COMMAND ----------

display(authDataSet)

# COMMAND ----------

# MAGIC %md ### Features

# COMMAND ----------

categorical = list(filter(lambda c: c.endswith("_CD"), authDataSet.columns))

numeric = list(filter(lambda c: not c.endswith("_CD"), authDataSet.columns))

# remove label
numeric.remove("label")

# remove double values we will impute
notimputed = [x for x in numeric if x not in dblCols]

print(categorical)
print(notimputed)

# COMMAND ----------

import copy
toImpute = copy.deepcopy(dblCols)
imputedCols = list(map(lambda i: i + "_IMP", toImpute))
list(zip(toImpute, imputedCols))

# COMMAND ----------

from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, StandardScaler

# categorical features
stringIndexers = list(map(lambda c: StringIndexer(inputCol = c, outputCol = c + "_IDX"), categorical))
catAssembler = VectorAssembler(inputCols = list(map(lambda c: c + "_IDX", categorical)), outputCol = 'catFeatures')

# numeric features
numericCols = imputedCols + notimputed
print(numericCols)

imputer = Imputer(strategy = 'median', inputCols = toImpute, outputCols = imputedCols)
numAssembler = VectorAssembler(inputCols = numericCols, outputCol='numericFeatures')
scaler = StandardScaler(inputCol = 'numericFeatures', outputCol = 'scaledFeatures' )

assembler = VectorAssembler(inputCols = ['catFeatures', 'scaledFeatures'], outputCol = 'features')

stages = stringIndexers + [catAssembler] + [imputer] + [numAssembler] + [scaler] + [assembler]

# COMMAND ----------

stages

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=stages)

featurizedDataset = pipeline.fit(authDataSet).transform(authDataSet)

# COMMAND ----------

display(featurizedDataset)

# COMMAND ----------

(train, test) = featurizedDataset.select('label', 'features').randomSplit([0.7, 0.3], seed=42)
train.cache()
test.cache()

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)

grid = ParamGridBuilder().addGrid(
    rf.maxDepth, [5, 10, 15]
).build()

ev = BinaryClassificationEvaluator()

cv = CrossValidator(estimator=rf, \
                    estimatorParamMaps=grid, \
                    evaluator=ev, \
                    numFolds=3)

cvModel = cv.fit(train)

# COMMAND ----------

predictions = cvModel.transform(test)

# COMMAND ----------

ev.setLabelCol("label").setMetricName("areaUnderROC")
print(ev.evaluate(predictions))

# COMMAND ----------

display(predictions)

# COMMAND ----------

total = authDataSet.select('*').count() * 1.0
fraud = authDataSet.select('*').where('label = 1.0').count()
nofraud = authDataSet.select('*').where('label = 0.0').count()
nofraud_pct = nofraud/total*100.0

print("fraud: {}".format(fraud))
print("no fraud: {}".format(nofraud))
print("total: {}".format(total))
print("fraud %: {0:.{1}f}".format(fraud/total*100, 4))
print("no fraud %: {0:.{1}f}".format(nofraud_pct, 4))

# COMMAND ----------

accuracy_pct = predictions.select('*').where('label = prediction').count() / (predictions.count() * 1.0) * 100
accuracy_pct

# COMMAND ----------

accuracy_pct - nofraud_pct

# COMMAND ----------

model = cvModel.bestModel

# COMMAND ----------

weights = map(lambda w: '%.3f' % w, model.featureImportances)
weightedFeatures = sorted(zip(categorical + numericCols, weights), key=lambda w: w[1], reverse=True)
weightedFeatures

# COMMAND ----------

spark.createDataFrame(weightedFeatures).toDF("feature", "weight").createOrReplaceTempView('wf')

# COMMAND ----------

# MAGIC %sql select feature, weight from wf where weight > .01 order by weight desc

# COMMAND ----------

# run once for each model build
# model.write().overwrite().save("/mnt/cc-fraud/model/gbt-model/")

# COMMAND ----------

# persist test schema for reading below
# test.schema.jsonValue()

# COMMAND ----------

# run once
# simulate streaming data source
# streamingData = test.repartition(600)
# streamingData.write.mode("overwrite").json("/mnt/cc-fraud/streaming/json/")

# COMMAND ----------

test.unpersist()
train.unpersist()

# COMMAND ----------

from pyspark.ml.classification import GBTClassificationModel
model = GBTClassificationModel.load("/mnt/cc-fraud/model/gbt-model/")

# COMMAND ----------

json = {'fields': [{'metadata': {},
   'type': 'double',
   'name': 'label',
   'nullable': True},
  {'metadata': {'ml_attr': {'num_attrs': 17,
     'attrs': {'numeric': [{'idx': 2, 'name': 'scaledFeatures_0'},
       {'idx': 3, 'name': 'scaledFeatures_1'},
       {'idx': 4, 'name': 'scaledFeatures_2'},
       {'idx': 5, 'name': 'scaledFeatures_3'},
       {'idx': 6, 'name': 'scaledFeatures_4'},
       {'idx': 7, 'name': 'scaledFeatures_5'},
       {'idx': 8, 'name': 'scaledFeatures_6'},
       {'idx': 9, 'name': 'scaledFeatures_7'},
       {'idx': 10, 'name': 'scaledFeatures_8'},
       {'idx': 11, 'name': 'scaledFeatures_9'},
       {'idx': 12, 'name': 'scaledFeatures_10'},
       {'idx': 13, 'name': 'scaledFeatures_11'},
       {'idx': 14, 'name': 'scaledFeatures_12'},
       {'idx': 15, 'name': 'scaledFeatures_13'},
       {'idx': 16, 'name': 'scaledFeatures_14'}],
      'nominal': [{'vals': ['90',
         '81',
         '1',
         '2',
         '0',
         '91',
         '82',
         '7',
         '5',
         '79'],
        'name': 'catFeatures_POS_ENTRY_MTHD_CD_IDX',
        'idx': 0},
       {'vals': ['0', '3', '5', '7', '4', '2', '1', '8'],
        'name': 'catFeatures_TRMNL_CLASFN_CD_IDX',
        'idx': 1}]}}},
   'type': {'sqlType': {'fields': [{'metadata': {},
       'type': 'byte',
       'name': 'type',
       'nullable': False},
      {'metadata': {}, 'type': 'integer', 'name': 'size', 'nullable': True},
      {'metadata': {},
       'type': {'containsNull': False,
        'type': 'array',
        'elementType': 'integer'},
       'name': 'indices',
       'nullable': True},
      {'metadata': {},
       'type': {'containsNull': False,
        'type': 'array',
        'elementType': 'double'},
       'name': 'values',
       'nullable': True}],
     'type': 'struct'},
    'type': 'udt',
    'pyClass': 'pyspark.ml.linalg.VectorUDT',
    'class': 'org.apache.spark.ml.linalg.VectorUDT'},
   'name': 'features',
   'nullable': True}],
 'type': 'struct'}

# COMMAND ----------

from pyspark.sql.types import StructType
schema = StructType.fromJson(json)

# COMMAND ----------

inputStreamDF = spark \
    .readStream \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .json("/mnt/cc-fraud/streaming/json/")

print(inputStreamDF.isStreaming)

# COMMAND ----------

scoredStream = model.transform(inputStreamDF).createOrReplaceTempView("stream_predictions")

# COMMAND ----------

# MAGIC %sql 
# MAGIC select prediction, count(1) as count
# MAGIC from stream_predictions 
# MAGIC group by prediction

# COMMAND ----------

