#you hav ebeen contracted by Hyundai  heavy industries to help them build a predictive model for some ships

# you have been flown out to ther HQ in Ulsan, South Korea

# its one of the worlds largest manufacturers of large ships, including cruise liners

# they need your help themgive accurate estimates of how many crew members a ship will require

# they are currently selling ships to some new customers and want you to create a model and use it to predict how many crew membersthe ship will need

# they mentioned that they have found  that particular cruise lines  will difer in acceptable crew counts , so it is most likely an important feature to include in your analysis

# cruise line is astring(need to convert to numbers ) hint- StringIndexer

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import  LinearRegression

spark = SparkSession.builder.appName('cruise').getOrCreate()
df = spark.read.csv('./data/cruise_ship_info.csv', inferSchema=True, header=True)

print "show the schema"
df.printSchema()

print "show the top 5 rows"
for ship in df.head(5):
    print ship
    print ""

print "see how "
df.groupBy("Cruise_line").count().show()


# give  the individual names
indexed = StringIndexer(inputCol="Cruise_line", outputCol='cruise_category')
indexed = indexed.fit(df).transform(df)
indexed.head(3)

print "this is the columns: {}".format(indexed.columns)

assembler = VectorAssembler(inputCols= ['Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density', 'crew', 'cruise_category'], outputCol ='features')

output  = assembler.transform(indexed)


output.select('features', 'crew').show()

final_data = output.select('features', 'crew')

train_data, test_data = final_data.randomSplit([0.7, 0.3])
# show training data 
train_data.describe().show()

# show testing data
test_data.describe().show()

# linear regression model
ship_lr = LinearRegression(  labelCol="crew")
trainned_ship_model = ship_lr.fit(train_data)

ship_results = trainned_ship_model.evaluate(test_data)
print "show root meansquared eerror: {}".format(ship_results.rootMeanSquaredError)

train_data.describe().show()

print "show this is r^2: {}".format(ship_results.r2 )

print "this is ship results"ship_results.residual)

# print "showing final data: "
# final_data.describe().show()

# @ section 11, lecture 39 and 13:12




