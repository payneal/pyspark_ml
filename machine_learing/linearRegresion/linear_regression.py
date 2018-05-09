from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
spark =  SparkSession.builder.appName('lrex').getOrCreate()

all_data = spark.read.format('libsvm').load(
    '.data/sample_linear_regression_data.txt')

#split data up 70/30
trainning, testing = all_data.randomSplit([0.7, 0.3])

print "look here"
trainning.show()
#  linear regression action
lr = LinearRegression(featuresCol="features", labelCol='label', predictionCol='prediction')
lrModel = lr.fit(trainning)
print 'model coefficients: {}'.format(lrModel.coefficients)
print "model intrecept: {}".format(lrModel.intercept)
trainning_summary = lrModel.summary
print "this is tainning summary functions: {}".format(
    dir(trainning_summary))
print "this is tainning summary root mean squared error: {}".format(
    trainning_summary.rootMeanSquaredError)


print "details for trainning data: "
trainning.describe().show()

print "details for testing data: "
testing.describe().show()

correct_model = lr.fit(trainning)

# evaluate is comparing our predictions to to the test data
test_results = correct_model.evaluate(testing)
test_results.residuals.show()
# print "this is rootMean Squared error: {}".format(test_results.rootMeanSquaredError)

unlabled_data = testing.select('features')
unlabled_data.show()

predictions = correct_model.transform(unlabled_data)
predictions.show()


