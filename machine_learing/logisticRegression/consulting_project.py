from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('logregconsult').getOrCreate()
data = spark.read.csv(
    './data/customer_churn.csv', inferSchema=True, header=True)
data.printSchema()
