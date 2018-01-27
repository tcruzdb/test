# Databricks notebook source
# MAGIC %md
# MAGIC #### 3. Advanced Analytics / ML
# MAGIC #### Impressions with clicks dataset
# MAGIC 
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/pub-tc/ML-workflow.png" width="800">

# COMMAND ----------

impression = spark.read \
  .parquet("/mnt/adtech/impression/parquet/train.csv/") \
  .selectExpr("*", "substr(hour, 7) as hr").repartition(200) # 4 workers, 64 cores

# COMMAND ----------

from pyspark.sql.functions import *

strCols = map(lambda t: t[0], filter(lambda t: t[1] == 'string', impression.dtypes))
intCols = map(lambda t: t[0], filter(lambda t: t[1] == 'int', impression.dtypes))

# [row_idx][json_idx]
strColsCount = sorted(map(lambda c: (c, impression.select(countDistinct(c)).collect()[0][0]), strCols), key=lambda x: x[1], reverse=True)
intColsCount = sorted(map(lambda c: (c, impression.select(countDistinct(c)).collect()[0][0]), intCols), key=lambda x: x[1], reverse=True)

# COMMAND ----------

# distinct counts for str columns
display(strColsCount)

# COMMAND ----------

# distinct counts for int columns
display(intColsCount)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

maxBins = 70
categorical = map(lambda c: c[0], filter(lambda c: c[1] <= maxBins, strColsCount))
categorical += map(lambda c: c[0], filter(lambda c: c[1] <= maxBins, intColsCount))
categorical.remove('click')

stringIndexers = map(lambda c: StringIndexer(inputCol = c, outputCol = c + "_idx"), categorical)
oneHotEncoders = map(lambda c: OneHotEncoder(inputCol = c + "_idx", outputCol = c + "_idx_ohe"), categorical)

assemblerInputs = map(lambda c: c + "_idx_ohe", categorical)
vectorAssembler = VectorAssembler(inputCols = assemblerInputs, outputCol = "features")

labelStringIndexer = StringIndexer(inputCol = "click", outputCol = "label")

stages = stringIndexers + oneHotEncoders + [vectorAssembler] + [labelStringIndexer]

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages = stages)

# create transformer to add features
featurizer = pipeline.fit(impression)

# COMMAND ----------

# dataframe with feature and intermediate transformation columns appended
featurizedImpressions = featurizer.transform(impression)

# COMMAND ----------

display(featurizedImpressions.select('features', 'label'))

# COMMAND ----------

train, test = featurizedImpressions \
  .select(["label", "features", "hr"]) \
  .randomSplit([0.7, 0.3], 42)
train.cache()
test.cache()

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

classifier = GBTClassifier(labelCol="label", featuresCol="features", maxBins=maxBins, maxDepth=10, maxIter=50)

model = classifier.fit(train)

# COMMAND ----------

predictions = model.transform(test)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

ev = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", metricName="areaUnderROC")
print ev.evaluate(predictions)

# COMMAND ----------

import json
features = map(lambda c: str(json.loads(json.dumps(c))['name']), \
               predictions.schema['features'].metadata.get('ml_attr').get('attrs').values()[0])
# convert numpy.float64 to str for spark.createDataFrame()
weights=map(lambda w: '%.10f' % w, model.featureImportances)
weightedFeatures = sorted(zip(weights, features), key=lambda x: x[1], reverse=True)
spark.createDataFrame(weightedFeatures).toDF("weight", "feature").createOrReplaceTempView('wf')

# COMMAND ----------

# MAGIC %sql 
# MAGIC select feature, weight 
# MAGIC from wf 
# MAGIC order by weight desc

# COMMAND ----------

predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql describe predictions

# COMMAND ----------

# MAGIC %sql select sum(case when prediction = label then 1 else 0 end) / (count(1) * 1.0) as accuracy
# MAGIC from predictions