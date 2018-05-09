from pyspark.sql import SparkSession
from pyspark.ml.regression import  LinearRegression
spark = SparkSession.builder.appName('lr_example').getOrCreate()


data = spark.read.csv('.data/Ecommerce_Customers.csv', inferSchema=True, header=True)

print "this is  the schema: "
data.printSchema()

print "this is the data: "
data.show()

print "just one entry: "
print data.head(1)

print "everything in row one: "
for item in data. head(1)[0]:
    print item


print "see all data columns: "
print data.columns

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols = ['Avg Session Length', 
        'Time on App', 'Time on Website', 'Length of Membership'],
    outpuCol ='features')

output = assembler.transform(data)

print "show new data frame"
output.printSchema()

print "show all details of row one data after wecreated vector of numbers: {}".format(
    output.head(1))

final_data = output.select('features', 'Yearly Amount Spent')

print "this is final data: "
final_data.show()

#  spliting the data
train, test = final_data.randomSplit([0.7, 0.3])

# show testing data
test.describe().show()

# show training data
train.describe().show()

# linnear regression model 
lr = LinearRegression( labelCol='Yearly Amount Spent')
lr_model = lr.fit(train)

test_results = lr_model.evaluate(test) 
test_results.residuals.show()

print "show root meansquared eerror: {}".format(test_results.rootMeanSquaredError)
print "show this is r^2: {}".format(test_results.r2 )

print "showing final data: "
final_data.describe().show()

print "key factors in this example: if you look at stddev its ~80 whilke rootmeanSquaredError is ~10, now if you look at r^2 result you cansee its alot closer to 1 than 0 infact its very close to 1 and if you include the previously mentioned one can tell this is a good model for this data"


print ""
print "now lets say we have all  data but yearly amaount spent and we want to fill it in:"

unlabled_data = test.select('features')

unlabled_data.show()

predictions = lr_model.transform(unlabled_data)
predictions.show()
