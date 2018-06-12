# Databricks notebook source
# MAGIC %md
# MAGIC #### Clinical diabetes data
# MAGIC ##### Given a set of clinical data labeled as `diabetes` or `not diabetes`, we create a model to classify new or unseen obervations.  This can help us understand attributes associated with diabetes.
# MAGIC 
# MAGIC ##### Diabetes dataset:
# MAGIC - ##### Attributes:
# MAGIC   * `pregnancies`
# MAGIC   * `plasma glucose`
# MAGIC   * `blood pressure`
# MAGIC   * `triceps skin thickness`
# MAGIC   * `insulin`
# MAGIC   * `bmi`
# MAGIC   * `diabetes pedigree`
# MAGIC   * `age`
# MAGIC - ##### Label:
# MAGIC   * `diabetes`
# MAGIC 
# MAGIC [Publicly available dataset](https://raw.githubusercontent.com/AvisekSukul/Regression_Diabetes/master/Custom%20Diabetes%20Dataset.csv)

# COMMAND ----------

# MAGIC %md 
# MAGIC %md ### ML Workflow
# MAGIC <img src="https://s3-us-west-2.amazonaws.com/pub-tc/ML-workflow.png" width="900" />

# COMMAND ----------

### run once
# %fs mkdirs /mnt/tcruz/diabetes

# COMMAND ----------

### run once
# %sh 
# wget "https://raw.githubusercontent.com/AvisekSukul/Regression_Diabetes/master/Custom%20Diabetes%20Dataset.csv"
# ls -lh
# mv -v "Custom Diabetes Dataset.csv" "custom_diabetes_dataset.csv"
# cp -v custom_diabetes_dataset.csv /dbfs/mnt/tcruz/diabetes
# ls -lh /dbfs/mnt/tcruz/diabetes

# COMMAND ----------

df = spark\
  .read\
  .option('header', True)\
  .option('inferSchema', True)\
  .csv('/mnt/tcruz/diabetes/custom_diabetes_dataset.csv')

# COMMAND ----------

df.count()

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.groupBy("diabetes").count().orderBy("diabetes"))

# COMMAND ----------

print("not diabetes: %.3f" % (df.where(df.diabetes == 0).count()/df.count()))
print("diabetes: %.3f" % (df.where(df.diabetes == 1).count()/df.count()))

# COMMAND ----------

help(df.corr)

# COMMAND ----------

sorted(zip(map(lambda c: df.corr('diabetes', c), df.columns), df.columns), reverse=True)

# COMMAND ----------

help(df.cov)

# COMMAND ----------

sorted(zip(map(lambda c: df.cov('diabetes', c), df.columns), df.columns), reverse=True)

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

# feature columns; remove 'diabetes' label
featureCols = df.columns
featureCols.remove('diabetes')

va = VectorAssembler(inputCols=featureCols, outputCol="va")
stages = [va, \
          StandardScaler(inputCol="va", outputCol="features", withStd=True, withMean=False), \
          StringIndexer(inputCol="diabetes", outputCol="label")]
pipeline = Pipeline(stages = stages)

featurizer = pipeline.fit(df)
featurizedDf = featurizer.transform(df)

# COMMAND ----------

display(featurizedDf)

# COMMAND ----------

train, test = featurizedDf.select(["label", "features"]).randomSplit([0.7, 0.3], 42)
train.cache()
test.cache()

# COMMAND ----------

display(train)

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)

# grid
grid = ParamGridBuilder()\
  .addGrid(rf.maxDepth, [5, 10, 20])\
  .addGrid(rf.numTrees, [20, 50, 100]).build()

# evaluator
ev = BinaryClassificationEvaluator()

# we use the featurized dataframe here; we'll use the full pipeline when we save the model
# we use estimator=rf to access the trained model to view feature importance
# stages += [rf]
# p = Pipeline(stages = stages)

# 10-fold cross validation
cv = CrossValidator(estimator=rf, \
                    estimatorParamMaps=grid, \
                    evaluator=ev, \
                    numFolds=10)

# cross-validated model
cvModel = cv.fit(train)

# COMMAND ----------

rf.explainParams()

# COMMAND ----------

predictions = cvModel.transform(test)

# COMMAND ----------

print(ev.evaluate(predictions))

# COMMAND ----------

cvModel.bestModel

# COMMAND ----------

display(test)

# COMMAND ----------

display(predictions)

# COMMAND ----------

cvModel.bestModel.featureImportances

# COMMAND ----------

print(cvModel.bestModel.getNumTrees)
print(cvModel.bestModel.toDebugString)

# COMMAND ----------

weights = map(lambda w: '%.10f' % w, cvModel.bestModel.featureImportances)
weightedFeatures = spark.createDataFrame(
  sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)
).toDF("weight", "feature")

# COMMAND ----------

display(weightedFeatures.select("feature", "weight").orderBy("weight", ascending=0))

# COMMAND ----------

weightedFeatures.createOrReplaceTempView('wf')

# COMMAND ----------

# MAGIC %sql 
# MAGIC select feature, weight 
# MAGIC from wf 
# MAGIC order by weight desc

# COMMAND ----------

# confusion matrix
display(predictions.groupBy("label", "prediction").count())

# COMMAND ----------

print("accuracy: %.3f" % (predictions.filter(predictions.label == predictions.prediction).count()/predictions.count()))