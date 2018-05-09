from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('dates').getOrCreate()
df = spark.read.csv(
    './data/appl_stock.csv', header=True, inferSchema=True)
print "starting  now ..."
print df.head(1)
print "showing again"
df.show()
print "showing date and open so you can look at time"
df.select(['Date', 'Open']).show()

from pyspark.sql.functions import (dayofmonth, hour, dayofyear, month, year, weekofyear, format_number, date_format)

print "show  all he dates"
df.select(dayofmonth(df['Date'])).show()

print "show all of the hours"
df.select(hour(df['Date'])).show()

print "show all of the months"
df.select(month(df['Date'])).show()

print "average closing price per year"
newdf = df.withColumn("Year", year(df['Date']))
# result = newdf.groupBy("Year").mean().select(["year", "avg(Close)"])
# newdf = result.withColumnRenamed("avg(Close)", "Average Closing price")

result = newdf.groupBy("Year").mean().select(["Year", "avg(Close)"])
new = result.withColumnRenamed("avg(Close)", "Average Closing Price")
new.select(['Year', format_number('Average Closing Price', 2).alias("Avg Close")]).show() 
