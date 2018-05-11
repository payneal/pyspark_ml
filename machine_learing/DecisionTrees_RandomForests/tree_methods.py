from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import (RandomForestClassifier, GBTClassifier, DecisionTreeClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName('myTree').getOrCreate()
data = spark.read.format('libsvm').load('./data/sample_libsvm_data.txt')
data.show()

# data split
train_data, test_data = data.randomSplit([0.7,0.3])

# create a decision tree classifier
# look a varibles like: maxDepth, maxBins, minInstancesPerNode, minInfoGain, etc ...
dtc =  DecisionTreeClassifier() #leaving default
# create a random forest classifier
# llok at varibles like: numTrees
rfc = RandomForestClassifier(numTrees=100)
# create gradient tree booster classifier
gbt = GBTClassifier()

# create models with classifications
dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)


# transform test data to get predictions
dtc_predictions =  dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)

print "look at decision tree predictions"
dtc_predictions.show()
print "look at random forest  predictions"
rfc_predictions.show()
print "look at gradient tree booster predictions"
gbt_predictions.show()

# using evaluator
dtc_accuracy_eval = MulticlassClassificationEvaluator(metricName='accuracy')
print "DTC ACCURACY: {}".format(dtc_accuracy_eval.evaluate(dtc_predictions)) # closer to 1 should raise eyebrows

rfc_accuracy_eval = MulticlassClassificationEvaluator(metricName='accuracy')
print "RFC ACCURACY: {}".format(rfc_accuracy_eval.evaluate(dtc_predictions)) # closer to 1 should raise eyebrows

gbt_accuracy_eval = MulticlassClassificationEvaluator(metricName='accuracy')
print "GBT ACCURACY: {}".format(gbt_accuracy_eval.evaluate(dtc_predictions)) # closer to 1 should raise eyebrows

# get the featureImportances for randomForestClassifier
# gives feature  and importance
print "rfc feature importances: {}".format(rfc_model.featureImportances)
