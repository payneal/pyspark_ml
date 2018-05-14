# background
# comapany has been recently hacked and need your help finding out about the hackers
# their forensic engineers have grabbed valuable data about the hack, inclusing information like session time, locations,
# wpm typing speed, etc.
# the forensic engineerrelates to you what she has been able to figure out so far, she has been able to grab meta-data of each session that the hackers used to connect to their servers
# these are the features of the data:
# "sessuib_connection_time": how long the session lasted in minutes
# "bytes_transfered": number of mb transferred durring session
# "kali_Trace_Used": indicates if the hacker was using Kali Linux
# "Servers_corrupted": NUmber of servers corrupted durring the attack
# "Pages_Corrupted": Numbers of pages illegally accessed
# "Location": Location attack came from (Probably useless because the hackers used VPNs
# "WPM_Typoing_speed": thier estimated typing speed based on session logs

# can you help figure out whether or not the third suspect had anything to do with the attacks, or was it just 2 hackers

# its probably not possible to know for sure, but maybe we can infer with clustering

# the forensic engineer knows that the hackers trad off attacks
# meaning they should each have roughly the same amount of attacks

# for ex if there were 100 total attacks, then in a 2 hacker situation each should have about 50 hacks, ina three hacker situation each wouild have about 33  hacks

# the engineer believes this is the key elements to solving this, but doesnt know how to distinguish this unlabled data into groups of hackers

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

spark = SparkSession.builder.appName('cluster').getOrCreate()

dataset = spark.read.csv('./data/hack_data.csv',  header=True, inferSchema=True)

print "here is the first row of data: {}".format(dataset.head())
print "here are the columns: {}".format(dataset.columns)


feat_col = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used', 'Servers_Corrupted', 'Pages_Corrupted', 'WPM_Typing_Speed']

assembler = VectorAssembler(inputCols=feat_col , outputCol="features")

final_data = assembler.transform(dataset)

final_data.show()

# practice  with scaling data (not needed I dont think)
scaler = StandardScaler(inputCol='features', outputCol="scaledFeatures")

scaler_model = scaler.fit(final_data)

cluster_final_data = scaler_model.transform(final_data)
cluster_final_data.printSchema()

# to see if it was one hacker or two we need to do kmeans for 2 ks and 3 ks

kmeans2 = KMeans(featuresCol="scaledFeatures", k=2)
kmeans3 = KMeans(featuresCol="scaledFeatures", k=3)

model_k2= kmeans2.fit(cluster_final_data)
model_k3= kmeans3.fit(cluster_final_data)

# hint abover said hackers alternated so they should have around the same total hacks

# checking guess of there being 3 hackers
print "3 hacker results: "
model_k3.transform(cluster_final_data).groupBy('prediction').count().show()

# checking guess of there being 2 hackers
print "2 hacker results: "
model_k2.transform(cluster_final_data).groupBy('prediction').count().show()

