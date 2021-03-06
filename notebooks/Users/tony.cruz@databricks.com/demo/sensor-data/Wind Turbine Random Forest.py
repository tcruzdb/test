# Databricks notebook source
# MAGIC %md #Wind Turbine Predictive Maintenance using Random Forest
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC Our dataset consists of vibration readings coming off sensors located in the gearboxes of wind turbines. We will use Random Forest Classification to predict which set of vibrations could be indicative of a failure.
# MAGIC 
# MAGIC *See image below for locations of the sensors*
# MAGIC 
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/databricks-demo-images/wind_turbine/wind_small.png" width=900 />
# MAGIC 
# MAGIC *Data Source Acknowledgement: This Data Source Provided By NREL*

# COMMAND ----------

# MAGIC %md ![Location](https://s3-us-west-2.amazonaws.com/databricks-demo-images/wind_turbine/wtsmall.png)

# COMMAND ----------

# MAGIC %sql use tcruz;

# COMMAND ----------

from pyspark.ml import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md ##1. ETL

# COMMAND ----------

# MAGIC %fs ls /windturbines/csv

# COMMAND ----------

# MAGIC %sh ls -lh /dbfs/windturbines/csv

# COMMAND ----------

schema = StructType([ \
    StructField("AN3", DoubleType(), False), \
    StructField("AN4", DoubleType(), False), \
    StructField("AN5", DoubleType(), False), \
    StructField("AN6", DoubleType(), False), \
    StructField("AN7", DoubleType(), False), \
    StructField("AN8", DoubleType(), False), \
    StructField("AN9", DoubleType(), False), \
    StructField("AN10", DoubleType(), False)  # , \
    # StructField("SPEED", DoubleType(), False) # , \ # speed format not consistent b/t classes
    # StructField("TORQUE", DoubleType(), False) \ # no torque data for healthy turbines
])
schema

# COMMAND ----------

damagedSensorReadings = spark.read.schema(schema).csv(
    "/windturbines/csv/D*gz").drop("TORQUE").drop("SPEED")
healthySensorReadings = spark.read.schema(schema).csv(
    "/windturbines/csv/H*gz").drop("TORQUE").drop("SPEED")

# COMMAND ----------

print
damagedSensorReadings.count()
print
healthySensorReadings.count()

# COMMAND ----------

# run once
healthySensorReadings.repartition(4).write.mode("overwrite").parquet(
    "/windturbines/healthy/parquet")
damagedSensorReadings.repartition(4).write.mode("overwrite").parquet(
    "/windturbines/damaged/parquet")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS tcruz.turbine_healthy (
# MAGIC   AN3 DOUBLE,
# MAGIC   AN4 DOUBLE,
# MAGIC   AN5 DOUBLE,
# MAGIC   AN6 DOUBLE,
# MAGIC   AN7 DOUBLE,
# MAGIC   AN8 DOUBLE,
# MAGIC   AN9 DOUBLE,
# MAGIC   AN10 DOUBLE
# MAGIC )
# MAGIC USING PARQUET
# MAGIC LOCATION '/windturbines/healthy/parquet'

# COMMAND ----------

# MAGIC %sql describe extended tcruz.turbine_healthy

# COMMAND ----------

# MAGIC %sql select count(1) from tcruz.turbine_healthy

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS tcruz.turbine_damaged (
# MAGIC   AN3 DOUBLE,
# MAGIC   AN4 DOUBLE,
# MAGIC   AN5 DOUBLE,
# MAGIC   AN6 DOUBLE,
# MAGIC   AN7 DOUBLE,
# MAGIC   AN8 DOUBLE,
# MAGIC   AN9 DOUBLE,
# MAGIC   AN10 DOUBLE
# MAGIC )
# MAGIC USING PARQUET
# MAGIC LOCATION '/windturbines/damaged/parquet'

# COMMAND ----------

# MAGIC %sql describe extended tcruz.turbine_damaged

# COMMAND ----------

# MAGIC %sql select count(1) from turbine_damaged

# COMMAND ----------

# MAGIC %md ##2. Data Exploration

# COMMAND ----------

turbine_damaged = table("tcruz.turbine_damaged")
turbine_healthy = table("tcruz.turbine_healthy")

# COMMAND ----------

print
turbine_healthy.count()
print
turbine_damaged.count()

# COMMAND ----------

display(turbine_healthy.describe())

# COMMAND ----------

display(turbine_damaged.describe())

# COMMAND ----------

randomSample = turbine_healthy.withColumn("ReadingType", lit("HEALTHY")).sample(
    False, 500 / 24000000.0) \
    .union(
    turbine_damaged.withColumn("ReadingType", lit("DAMAGED")).sample(False,
                                                                     500 / 24000000.0)).cache()

# COMMAND ----------

display(randomSample)

# COMMAND ----------

display(randomSample)

# COMMAND ----------

display(randomSample)

# COMMAND ----------

display(randomSample)

# COMMAND ----------

display(randomSample)

# COMMAND ----------

# MAGIC %md ##3. Model Creation And Export

# COMMAND ----------

# MAGIC %md ### Workflows with Pyspark.ML Pipeline
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/pub-tc/ML-workflow.png" width="900">

# COMMAND ----------

df = turbine_healthy.withColumn("ReadingType", lit("HEALTHY")) \
    .union(turbine_damaged.withColumn("ReadingType", lit("DAMAGED")))

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
va = VectorAssembler(inputCols=featureCols, outputCol="va")
stages = [va, \
          StandardScaler(inputCol="va", outputCol="features", withStd=True,
                         withMean=True), \
          StringIndexer(inputCol="ReadingType", outputCol="label")]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

featurizer = pipeline.fit(df)
featurizedDf = featurizer.transform(df)

# COMMAND ----------

display(featurizedDf)

# COMMAND ----------

train, test = featurizedDf.select(["label", "features"]).randomSplit([0.7, 0.3],
                                                                     42)
train = train.repartition(32)
train.cache()
test.cache()
print
train.count()
print
train.rdd.getNumPartitions()

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)

# grid
grid = ParamGridBuilder().addGrid(
    rf.maxDepth, [4, 5, 6]
).build()

# evaluator
ev = BinaryClassificationEvaluator()

# we use the featurized dataframe here; we'll use the full pipeline when we save the model
# we use estimator=rf to access the trained model to view feature importance
# stages += [rf]
# p = Pipeline(stages = stages)

# 3-fold cross validation
cv = CrossValidator(estimator=rf, \
                    estimatorParamMaps=grid, \
                    evaluator=ev, \
                    numFolds=3)

# cross-validated model
cvModel = cv.fit(train)

# COMMAND ----------

predictions = cvModel.transform(test)

# COMMAND ----------

cvModel

# COMMAND ----------

cvModel.bestModel

# COMMAND ----------

# MAGIC %md 
# MAGIC <img src="https://www.researchgate.net/profile/Mauno_Vihinen/publication/230614354/figure/fig4/AS:216471646019585@1428622270943/Contingency-matrix-and-measures-calculated-based-on-it-2x2-contigency-table-for.png" width=600 />
# MAGIC <img src="https://www.medcalc.org/manual/_help/images/roc_intro3.png" width=400 />

# COMMAND ----------

# display(cvModel.bestModel, featurizedDf, "ROC")
# not yet supported

# COMMAND ----------

print
ev.evaluate(predictions)

# COMMAND ----------

bestModel = cvModel.bestModel
print
bestModel
print
'\n' + str(bestModel.getNumTrees)
print
'\n' + bestModel.explainParams()
print
'\n' + bestModel.toDebugString

# COMMAND ----------

cvModel.bestModel.featureImportances

# COMMAND ----------

# convert numpy.float64 to str for spark.createDataFrame()
weights = map(lambda w: '%.10f' % w, bestModel.featureImportances)
weightedFeatures = spark.createDataFrame(
    sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)
).toDF("weight", "feature")

# COMMAND ----------

display(
    weightedFeatures.select("feature", "weight").orderBy("weight", ascending=0))

# COMMAND ----------

weightedFeatures.createOrReplaceTempView('wf')

# COMMAND ----------

# MAGIC %sql 
# MAGIC select feature, weight 
# MAGIC from wf 
# MAGIC order by weight desc

# COMMAND ----------

predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql select
# MAGIC sum(case when prediction = label then 1 else 0 end) / (count(1) * 1.0) as pct_match
# MAGIC from predictions

# COMMAND ----------

stages += [rf]
stages

# COMMAND ----------

train, foo = df.randomSplit([0.7, 0.3], 42)
train = train.repartition(32)
train.cache()

# COMMAND ----------

cv = CrossValidator(estimator=Pipeline(stages=stages), \
                    estimatorParamMaps=grid, \
                    evaluator=BinaryClassificationEvaluator(), \
                    numFolds=3)

cvModel = cv.fit(train)

# COMMAND ----------

cvModel.bestModel

# COMMAND ----------

cvModel.bestModel.write().overwrite().save("/mnt/tcruz/windTurbine/model/")

# COMMAND ----------

# MAGIC %sh ls /dbfs/mnt/tcruz/windTurbine/model/

# COMMAND ----------

model = PipelineModel.load("/mnt/tcruz/windTurbine/model/")

# COMMAND ----------

# run once
# simulate streaming data source
# streamingData = df.repartition(300)
# print streamingData.rdd.getNumPartitions()
# streamingData.write.mode("overwrite").json("/mnt/tcruz/windTurbine/streaming-json/")

# COMMAND ----------

# MAGIC %sh ls /dbfs/mnt/tcruz/windTurbine/streaming-json/

# COMMAND ----------

inputStream = spark \
    .readStream \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .json("/mnt/tcruz/windTurbine/streaming-json/")

# COMMAND ----------

scoredStream = model.transform(inputStream)

# COMMAND ----------

scoredStream.writeStream \
    .format("memory") \
    .queryName("stream") \
    .start()

# COMMAND ----------

# MAGIC %sql select prediction, count(1) from stream group by prediction

# COMMAND ----------

# MAGIC %sql describe stream

# COMMAND ----------

# MAGIC %sql select prediction, avg(AN9), min(AN9), max(AN9) from stream group by prediction

# COMMAND ----------
