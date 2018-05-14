from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

spark = SparkSession.builder.appName('cluster').getOrCreate()
dataset = spark.read.csv('./data/seeds_dataset.csv', header= True, inferSchema=True)
dataset.printSchema()

# show  firs row
print dataset.head(1)

# domain knowlege tells us 3 type of seed types 


# lets look at all the columns
print "here are all the columns"
print dataset.columns

assembler = VectorAssembler(inputCols=dataset.columns, outputCol='features') 

final_data = assembler.transform(dataset)

final_data.printSchema()

# scaling the data
scaler = StandardScaler(inputCol="features", outputCol='scaledFeatures')

scaler_model = scaler.fit(final_data)
final_data = scaler_model.transform(final_data)

#  viewing final data
print final_data.head(1)

kmeans = KMeans(featuresCol="scaledFeatures", k=3)
model = kmeans.fit(final_data)

print "WSSSE: "
print  model.computeCost(final_data )

print "centers: "
centers = model.clusterCenters()
print centers

# get predictions
model.transform(final_data).select('prediction').show()
