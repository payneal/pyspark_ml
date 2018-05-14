from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName('cluster').getOrCreate()
data_set = spark.read.format('libsvm').load("./data/sample_kmeans_data.txt")

# show all of the data
data_set.show()

# to show you that you only need features will get rid of lables
final_data = data_set.select('features')

# show final data
final_data.show()

# creating model
kmeans = KMeans().setK(3).setSeed(1)

# fit the model
model = kmeans.fit(final_data )

# evaluate clustering algo
# wssse =  within set sum of squared errors
wssse = model.computeCost(final_data)
print "this is within set summ squared errors: {}".format(wssse)

# check out final data
final_data.show()

# see centers for clustering
centers = model.clusterCenters()

print "these are the centroids: {}".format(centers)
# see line 17 we set to two so there will be two centers

# transform your data
results = model.transform(final_data)

# show results
results.show()

