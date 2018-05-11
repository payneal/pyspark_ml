from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
spark=SparkSession.builder.appName('logregconsult').getOrCreate()
data = spark.read.csv(
    './data/customer_churn.csv', inferSchema=True, header=True)
# check out schema
data.printSchema()
# see more info about about data frame
data.describe().show()
# checl out columns
data.columns
# select stuff for vector assembler
assembler = VectorAssembler(inputCols=["Age", "Total_Purchase", "Account_Manager", "Years", "Num_Sites"],  outputCol="features" )
# take vector and transform the data
output = assembler.transform(data)
# create a new data frame with only verctor of features(see line 14) and churn
final_data = output.select('features', 'churn')
# split the data (train and test) with 70/30 split 
train_churn, test_churn = final_data.randomSplit([0.7,0.3])

# create a logistic regression
lr_churn = LogisticRegression(labelCol='churn')

# create a data frame from the logistic regression that is trained w/ train churn
fitted_churn_model = lr_churn.fit(train_churn)

# check out a summary of the data
trainning_sum = fitted_churn_model.summary

# call the predictions method to check things out
trainning_sum.predictions.describe().show()

# get the predictions and lables
pred_and_labels = fitted_churn_model.evaluate(test_churn)

# show the data frame predictions
pred_and_labels.predictions.show()

churn_eval = BinaryClassificationEvaluator(rawPredictionCol= "prediction", labelCol="churn")

# area under the curve
auc = churn_eval.evaluate(pred_and_labels.predictions)
print "this is auc(if its above .50 means then its better than random): {}".format(auc)


# predict on new data: (training this on all final data (no split))
final_lr_model = lr_churn.fit(final_data)\

# pull in new customers data
new_customers = spark.read.csv('data/new_customers.csv', inferSchema=True, header=True)

# show the new customers
new_customers.printSchema()

test_new_customers = assembler.transform(new_customers)
test_new_customers.printSchema()

final_results = final_lr_model.transform(test_new_customers)

# using created data form to make predictions on new customer data 
final_results.select('Company','prediction').show()

#  show test_new_customers info
test_new_customers.describe().show()

