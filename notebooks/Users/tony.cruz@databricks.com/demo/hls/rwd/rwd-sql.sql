-- Databricks notebook source
-- MAGIC %md
-- MAGIC <img src="https://assets.cms.gov/resources/cms/images/logo/site-logo.png" />
-- MAGIC 
-- MAGIC ### [CMS Dataset Downloads](https://www.cms.gov/OpenPayments/Explore-the-Data/Dataset-Downloads.html): General Payments 2016
-- MAGIC 
-- MAGIC * `OP_DTL_GNRL_PGYR2016_P01172018.csv: 
-- MAGIC This file contains the data set of General Payments reported for the 2016 program year.`

-- COMMAND ----------

-- MAGIC %md
-- MAGIC <img src="https://s3-us-west-2.amazonaws.com/pub-tc/img-fraud_ml_pipeline.png" alt="workflow" width="500">

-- COMMAND ----------

use cms

-- COMMAND ----------

select recipient_state, count(1) as count
from cms.gp2016geo
where length(recipient_state) = 2
group by 1
order by 2 desc

-- COMMAND ----------

select recipient_state, count(1) as count
from cms.gp2016geo
where length(recipient_state) = 2
group by 1
order by 2 desc

-- COMMAND ----------

select recipient_state, sum(cast(total_amount_of_payment_usdollars as double)) as total
from cms.gp2016geo
where total_amount_of_payment_usdollars is not null
group by 1 order by 2 desc

-- COMMAND ----------

select recipient_state, sum(cast(total_amount_of_payment_usdollars as double)) as total
from cms.gp2016geo
where total_amount_of_payment_usdollars is not null
group by 1 order by 2 desc

-- COMMAND ----------

select recipient_city, recipient_state, sum(cast(total_amount_of_payment_usdollars as double)) as total
from cms.gp2016
where total_amount_of_payment_usdollars is not null
group by 1,2 order by 3 desc

-- COMMAND ----------

select Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1, count(1) as count, sum(cast(total_amount_of_payment_usdollars as double)) as total
from cms.gp2016
where Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1 is not null
group by 1 order by 3 desc

-- COMMAND ----------

select Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1, count(1) as count
from cms.gp2016
where Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1 = 'COUMADIN'
group by 1 order by 1

-- COMMAND ----------

select Product_Category_or_Therapeutic_Area_1, count(1) as count, sum(cast(total_amount_of_payment_usdollars as double)) as total
from cms.gp2016
where Product_Category_or_Therapeutic_Area_1 is not null
group by 1 order by 3 desc

-- COMMAND ----------

select Name_of_Third_Party_Entity_Receiving_Payment_or_Transfer_of_Value, sum(cast(total_amount_of_payment_usdollars as double)) as total
from cms.gp2016
where Name_of_Third_Party_Entity_Receiving_Payment_or_Transfer_of_Value is not null
group by 1 order by 2 desc

-- COMMAND ----------

select Name_of_Third_Party_Entity_Receiving_Payment_or_Transfer_of_Value, sum(cast(total_amount_of_payment_usdollars as double)) as total
from cms.gp2016
where Name_of_Third_Party_Entity_Receiving_Payment_or_Transfer_of_Value is not null
group by 1 order by 2 desc
limit 100

-- COMMAND ----------

select Associated_Drug_or_Biological_NDC_1, count(1)
from cms.gp2016
where Associated_Drug_or_Biological_NDC_1 is not null
group by 1
order by 2 desc

-- COMMAND ----------

