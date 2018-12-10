# Databricks notebook source
# MAGIC %md ## Credit Card Fraud Prediction
# MAGIC 
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/pub-tc/ML-workflow.png" width="800">
# MAGIC 
# MAGIC #### 1. Load data
# MAGIC #### 2. Explore data, features
# MAGIC #### 3. Create ML model / Evaluate
# MAGIC 
# MAGIC Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud/home
# MAGIC 
# MAGIC The datasets contains transactions made by credit cards in September 2013 by european cardholders.
# MAGIC 
# MAGIC Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

# COMMAND ----------

# MAGIC %sh ls -lh /dbfs/mnt/cc-fraud-2/data/creditcard.csv

# COMMAND ----------

# MAGIC %fs head /mnt/cc-fraud-2/data/creditcard.csv

# COMMAND ----------

df = spark.read.options(header='True', inferSchema='True').csv('/mnt/cc-fraud-2/data/creditcard.csv')
print(df.rdd.getNumPartitions())
print(df.count())

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# no na values
print("NA values? " + str(df.dropna().count() != df.count()))

# COMMAND ----------

intCols = list(map(lambda t: t[0], filter(lambda t: t[1] == 'int', df.dtypes)))
dblCols = list(map(lambda t: t[0], filter(lambda t: t[1] == 'double', df.dtypes)))

print(intCols)
print(dblCols)

# COMMAND ----------

from pyspark.sql.functions import *
# this can give us an idea of categorical vs continuous (numeric) columns
# [row_idx][json_idx]
intColsCount = sorted(map(lambda c: (c, df.select(countDistinct(c)).collect()[0][0]), intCols), key=lambda kv: kv[1], reverse=True)
dblColsCount = sorted(map(lambda c: (c, df.select(countDistinct(c)).collect()[0][0]), dblCols), key=lambda kv: kv[1], reverse=True)

# COMMAND ----------

display(intColsCount)

# COMMAND ----------

display(dblColsCount)

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.describe())

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

stages = [VectorAssembler(inputCols=dblCols, outputCol="va"), \
          StandardScaler(inputCol="va", outputCol="features"), \
          StringIndexer(inputCol="Class", outputCol="label")]

pipeline = Pipeline(stages=stages)

# COMMAND ----------

featurizedDf = pipeline.fit(df).transform(df)

# COMMAND ----------

display(featurizedDf)

# COMMAND ----------

total = featurizedDf.select('*').count() * 1.0
fraud = featurizedDf.select('*').where('label = 1').count()
nofraud = featurizedDf.select('*').where('label = 0').count()
nofraud_pct = nofraud/total*100.0

print("fraud: {}".format(fraud))
print("no fraud: {}".format(nofraud))
print("total: {}".format(total))
print("fraud %: {0:.{1}f}".format(fraud/total*100, 4))
print("no fraud %: {0:.{1}f}".format(nofraud_pct, 4))

# COMMAND ----------

train, test = featurizedDf.select(["label", "features"]).randomSplit([0.7, 0.3], 42)
train = train.repartition(16)
train.cache()
test.cache()
print(train.count())
print(test.count())

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=42)

# grid
grid = ParamGridBuilder().addGrid(
    gbt.maxDepth, [4, 5, 6]
).build()

# evaluator
ev = BinaryClassificationEvaluator()

# we use estimator=gbt to access the trained model to view feature importance

# 3-fold cross validation
cv = CrossValidator(estimator=gbt, \
                    estimatorParamMaps=grid, \
                    evaluator=ev, \
                    numFolds=3)

# cross-validated model
cvModel = cv.fit(train)

# COMMAND ----------

predictions = cvModel.transform(test)

# COMMAND ----------

ev.setLabelCol("label").setMetricName("areaUnderROC")
print(ev.evaluate(predictions))

# COMMAND ----------

# highly skewed dataset
ev.setLabelCol("label").setMetricName("areaUnderPR")
print(ev.evaluate(predictions))

# COMMAND ----------

accuracy_pct = predictions.select('*').where('label = prediction').count() / (predictions.count() * 1.0) * 100
print(str(accuracy_pct) + ": accuracy")
print(str(accuracy_pct - nofraud_pct) + ": improvement over baseline")

# COMMAND ----------

model = cvModel.bestModel
model

# COMMAND ----------

weights = map(lambda w: '%.3f' % w, model.featureImportances)
weightedFeatures = sorted(zip(dblCols, weights), key=lambda w: w[1], reverse=True)
weightedFeatures

# COMMAND ----------

display(spark.createDataFrame(weightedFeatures).toDF("feature", "weight").select("feature", "weight").where("weight > 0.0"))

# COMMAND ----------

# run once for each model build
model.write().overwrite().save("/mnt/cc-fraud-2/model/gbt-model/")

# COMMAND ----------

# persist test schema for reading below
# test.schema.jsonValue()

# COMMAND ----------

# run once
# simulate streaming data source
# streamingData = test.repartition(500)
# streamingData.write.mode("overwrite").json("/mnt/cc-fraud-2/streaming/json/")

# COMMAND ----------

test.unpersist()
train.unpersist()

# COMMAND ----------

from pyspark.ml.classification import GBTClassificationModel
model = GBTClassificationModel.load("/mnt/cc-fraud-2/model/gbt-model/")

# COMMAND ----------

json = {'fields': [{'metadata': {u'ml_attr': {u'name': u'label',
     u'type': u'nominal',
     u'vals': [u'0', u'1']}},
   'name': 'label',
   'nullable': False,
   'type': 'double'},
  {'metadata': {},
   'name': 'features',
   'nullable': True,
   'type': {'class': 'org.apache.spark.ml.linalg.VectorUDT',
    'pyClass': 'pyspark.ml.linalg.VectorUDT',
    'sqlType': {'fields': [{'metadata': {},
       'name': 'type',
       'nullable': False,
       'type': 'byte'},
      {'metadata': {}, 'name': 'size', 'nullable': True, 'type': 'integer'},
      {'metadata': {},
       'name': 'indices',
       'nullable': True,
       'type': {'containsNull': False,
        'elementType': 'integer',
        'type': 'array'}},
      {'metadata': {},
       'name': 'values',
       'nullable': True,
       'type': {'containsNull': False,
        'elementType': 'double',
        'type': 'array'}}],
     'type': 'struct'},
    'type': 'udt'}}],
 'type': 'struct'}

# COMMAND ----------

from pyspark.sql.types import StructType
schema = StructType.fromJson(json)

# COMMAND ----------

inputStreamDF = spark \
    .readStream \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .json("/mnt/cc-fraud-2/streaming/json/")

print(inputStreamDF.isStreaming)

# COMMAND ----------

scoredStream = model.transform(inputStreamDF).createOrReplaceTempView("stream_predictions")

# COMMAND ----------

# MAGIC %sql 
# MAGIC select prediction, count(1) as count
# MAGIC from stream_predictions 
# MAGIC group by prediction

# COMMAND ----------

