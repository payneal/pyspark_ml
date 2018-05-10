from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (BinaryClassificationEvaluator, MulticlassClassificationEvaluator)
spark = SparkSession.builder.appName('mylogre').getOrCreate()

my_data = spark.read.format("libsvm").load('./data/sample_libsvm_data.txt')
print "here is sample data already in vector format"
my_data.show()

# setting up logistic regression model
my_log_reg_model= LogisticRegression()

# fit data to logistic  regression
fitted_logreg = my_log_reg_model.fit(my_data)

# get summary ooff of data fitted to loigistic regression
log_summary = fitted_logreg.summary

# get predictions
log_summary.predictions.printSchema()

# show data frame of log summary
log_summary.predictions.show()


# random split on data
lr_trian, lr_test =  my_data.randomSplit([0.7, 0.3])

# retain then evaluate on  trainning data
final_model = LogisticRegression()
final_model = final_model.fit(lr_train)

prediction_and_labels =  fit_final.evaluate(lr_test)
prediction_and_labels.predictions.show()

# using binary  classication evaluatior
my_eval = BinaryClassificationEvaluator()
my_final_roc = my_eval.evaluate(prediction_and_labels.predictions)

print "this is final_roc: {}".format(my_final_roc)
