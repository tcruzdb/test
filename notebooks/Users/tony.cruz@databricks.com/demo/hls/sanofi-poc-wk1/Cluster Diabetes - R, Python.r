# Databricks notebook source
# MAGIC %md
# MAGIC #### Clustering clinical diabetes data using k-means
# MAGIC ##### Given a set of clinical data labeled as `diabetes` or `not diabetes`, we create a model using k-means to partition new or unseen obervations.  This can help us understand clinical data that may serve as predictors of diabetes.
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

# MAGIC %md #### 1. We will create a bisecting k-means model using SparkR

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

library(SparkR)

# COMMAND ----------

df <- read.df("/mnt/tcruz/diabetes/custom_diabetes_dataset.csv", source = "csv", header = "true", inferSchema = "true")

# COMMAND ----------

display(df)

# COMMAND ----------

printSchema(df)

# COMMAND ----------

display(summary(df))

# COMMAND ----------

count(df)

# COMMAND ----------

split = randomSplit(df, c(7,3))
train = split[[1]]
test = split[[2]]

# COMMAND ----------

display(train)

# COMMAND ----------

count(train)

# COMMAND ----------

count(test)

# COMMAND ----------

?spark.bisectingKmeans

# COMMAND ----------

?fitted

# COMMAND ----------

?formula

# COMMAND ----------

# Fit bisecting k-means model with 2 centers
# R formula: diabetes ~ .
model <- spark.bisectingKmeans(train, diabetes ~ .,  k = 2, seed = 42)

# returns training set with predictions
fitted.result <- fitted(model, "centers")

display(fitted.result)

# COMMAND ----------

summary(model)

# COMMAND ----------

summary(model)$coefficients

# COMMAND ----------

summary(model)$size

# COMMAND ----------

?predict

# COMMAND ----------

predictions <- predict(model, test)

# COMMAND ----------

display(predictions)

# COMMAND ----------

display(select(predictions, "insulin", "diabetes", "prediction"))

# COMMAND ----------

createOrReplaceTempView(predictions, "predictions")

# COMMAND ----------

# MAGIC %sql describe predictions

# COMMAND ----------

# MAGIC %sql select
# MAGIC sum(case when prediction = diabetes then 1 else 0 end) / (count(1) * 1.0) as pct_match
# MAGIC from predictions

# COMMAND ----------

# MAGIC %sql select prediction, avg(insulin), min(insulin), max(insulin) from predictions group by prediction order by prediction

# COMMAND ----------

display(predictions)

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Python k-means clustering diabetic and non-diabetic patients based on clinical parameters.
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-roy/KM.jpg" width=600 />

# COMMAND ----------

# MAGIC %md #### Python in an R notebook: the `%python` magic command

# COMMAND ----------

# MAGIC %python
# MAGIC df = spark.read.option('header', True).option('inferSchema', True).csv('/mnt/tcruz/diabetes/custom_diabetes_dataset.csv')

# COMMAND ----------

# MAGIC %python
# MAGIC df.count()

# COMMAND ----------

# MAGIC %python
# MAGIC display(df)

# COMMAND ----------

# MAGIC %python
# MAGIC help(df.corr)

# COMMAND ----------

# MAGIC %python
# MAGIC sorted(zip(map(lambda c: df.corr('diabetes', c), df.columns), df.columns), reverse=True)

# COMMAND ----------

# MAGIC %python
# MAGIC help(df.cov)

# COMMAND ----------

# MAGIC %python
# MAGIC sorted(zip(map(lambda c: df.cov('diabetes', c), df.columns), df.columns), reverse=True)

# COMMAND ----------

# MAGIC %python
# MAGIC display(df)

# COMMAND ----------

# MAGIC %python
# MAGIC from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
# MAGIC from pyspark.ml import Pipeline
# MAGIC 
# MAGIC # feature columns; we don't include 'diabetes' here since that is the label
# MAGIC featureCols = df.columns
# MAGIC featureCols.remove('diabetes')
# MAGIC 
# MAGIC va = VectorAssembler(inputCols=featureCols, outputCol="va")
# MAGIC stages = [va, \
# MAGIC           StandardScaler(inputCol="va", outputCol="features", withStd=True, withMean=False), \
# MAGIC           StringIndexer(inputCol="diabetes", outputCol="label")]
# MAGIC pipeline = Pipeline(stages = stages)
# MAGIC 
# MAGIC featurizer = pipeline.fit(df)
# MAGIC featurizedDf = featurizer.transform(df)

# COMMAND ----------

# MAGIC %python
# MAGIC display(featurizedDf)

# COMMAND ----------

# MAGIC %python
# MAGIC train, test = featurizedDf.select(["label", "features"]).randomSplit([0.7, 0.3], 42)
# MAGIC train.cache()
# MAGIC test.cache()

# COMMAND ----------

# MAGIC %python
# MAGIC display(train)

# COMMAND ----------

# MAGIC %python
# MAGIC from pyspark.ml.clustering import KMeans
# MAGIC kmeans = KMeans().setK(2).setSeed(42)
# MAGIC model = kmeans.fit(train)

# COMMAND ----------

# MAGIC %python
# MAGIC # wssse
# MAGIC print 'cost: ' + str(model.computeCost(train))
# MAGIC print '\ncenters: ' + str(model.clusterCenters())
# MAGIC print '\ncluster sizes: ' + str(model.summary.clusterSizes)

# COMMAND ----------

# MAGIC %python
# MAGIC display(test)

# COMMAND ----------

# MAGIC %python
# MAGIC predictions = model.transform(test)

# COMMAND ----------

# MAGIC %python
# MAGIC display(predictions)

# COMMAND ----------

# MAGIC %python
# MAGIC # Training set
# MAGIC # K-means Clustering: Visualizing Clusters
# MAGIC # https://databricks.com/blog/2015/10/27/visualizing-machine-learning-models.html
# MAGIC 
# MAGIC # 0 "pregnancies"
# MAGIC # 1 "plasma glucose"
# MAGIC # 2 "blood pressure"
# MAGIC # 3 "triceps skin thickness"
# MAGIC # 4 "insulin"
# MAGIC # 5 "bmi"
# MAGIC # 6 "diabetes pedigree"
# MAGIC # 7 "age"
# MAGIC 
# MAGIC display(model, train)

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC # Test set
# MAGIC # K-means Clustering: Visualizing Clusters
# MAGIC # https://databricks.com/blog/2015/10/27/visualizing-machine-learning-models.html
# MAGIC 
# MAGIC # 0 "pregnancies"
# MAGIC # 1 "plasma glucose"
# MAGIC # 2 "blood pressure"
# MAGIC # 3 "triceps skin thickness"
# MAGIC # 4 "insulin"
# MAGIC # 5 "bmi"
# MAGIC # 6 "diabetes pedigree"
# MAGIC # 7 "age"
# MAGIC 
# MAGIC display(model, test)

# COMMAND ----------

