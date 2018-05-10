from pyspark.sql import SparkSession
spark = SparkSession.builder.appName(
    'titantic').getOrCreate()
df = spark.read.csv("./data/titanic.csv")
df.printSchema()


