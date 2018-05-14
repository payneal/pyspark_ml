from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.appName('tree_consult').getOrCreate()

data = spark.read.csv('./data/dog_food.csv', inferSchema=True, header=True)
 
# show first row of collected data
print "this is first row of data: {}".format(data.head(1))

# show all of the columns
print "all of columns:  {}".format(data.columns)

# create an assembler object
assembler = VectorAssembler(inputCols=['A', 'B','C', 'D'], outputCol='features')

# transform the data
output = assembler.transform(data)

# setting up randomForest  classifier(
rfc = RandomForestClassifier(labelCol="Spoiled",  featuresCol="features")

#  check out the schema:
output.printSchema()

# only need features and spoiled col
final_data = output.select(['features', 'Spoiled'])

# look at final_data
final_data.show()

# training the classifier
rfc_model = rfc.fit(final_data)

# check out the first row 
print "first row of data: {}".format(final_data.head(1))

# now that we have trained our data we have the ablilty to check out feature importance(its basedon index)
# thais is why we printed the first line ofteh data so you could compare results

print "feature Importances: {}".format(rfc_model.featureImportances)

# if you look at the result its seemsas if C has 90%+ importance on weather its spoiled or not


# no need to train data or anything like that we are just discovering which col/varible had the more influnce or  lead to causation to the identified co/varible we examined
