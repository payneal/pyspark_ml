# workthrough all 3 tree methofd  discussed  and  compare their results on a college dataset 
# the data set has features of universities and labled either Private or Public

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import (RandomForestClassifier, GBTClassifier, DecisionTreeClassifier)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('tree').getOrCreate()
data  = spark.read.csv('./data/College.csv', inferSchema=True, header=True)

# get idea what data looks like
data.printSchema()

# check out the first entry of data 
print data.head(1)
print "here are columns: {}".format(data.columns)

assembler = VectorAssembler(
    inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc',
	'F_Undergrad', 'P_Undergrad', 'Outstate', 'Room_Board', 'Books',
	'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni', 'Expend', 'Grad_Rate'],
    outputCol='features')

output = assembler.transform(data)

# trying to predicit if a school is private or not
indexer = StringIndexer(inputCol='Private', outputCol='PrivateIndex')

output_fixed = indexer.fit(output).transform(output)
output_fixed.printSchema()

final_data = output_fixed.select('features', 'PrivateIndex')
train_data, test_data = final_data.randomSplit([0.7,0.3])

dtc = DecisionTreeClassifier(labelCol='PrivateIndex', featuresCol='features')
rfc = RandomForestClassifier(numTrees=150, labelCol='PrivateIndex', featuresCol='features')
gbt = GBTClassifier(labelCol='PrivateIndex', featuresCol='features')

# fitting train data
dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)

# get predictions 
dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)

my_binary_eval = BinaryClassificationEvaluator(labelCol="PrivateIndex")

print "Decision Tree Classification: "
print my_binary_eval.evaluate(dtc_preds)
print "Random Forest Classification: "
print my_binary_eval.evaluate(rfc_preds)


my_binary_eval_2 = BinaryClassificationEvaluator(
	labelCol="PrivateIndex", rawPredictionCol='prediction')
print "Gradient tree booster: "
print my_binary_eval_2.evaluate(gbt_preds)

acc_eval =MulticlassClassificationEvaluator(
	labelCol='PrivateIndex',  metricName='accuracy')

rfc_acc = acc_eval.evaluate(rfc_preds)

print "this is the new rfc_acc: {}".format(rfc_acc)
 
