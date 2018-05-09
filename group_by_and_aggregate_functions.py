from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('aggs').getOrCreate()
df = spark.read.csv(
    "./data/sales_info.csv", inferSchema=True, header=True)

print "showing the show function"
df.show()

print "below is the printSchema functionality"
df.printSchema()

# more examples of grouping can be sceen in docs

print "example of group by mean"
df.groupBy('Company').mean().show()

print "example of group by sum"
df.groupBy('Company').sum().show()

print "example of group by max" 
df.groupBy('Company').max().show()

print "example of group by min"
df.groupBy('Company').min().show()

print "example of how many rows there per company"
df.groupBy('Company').count().show()

# aggreate examples
print "get the sum of sales: "
df.agg({'Sales':'sum'}).show()

print "get the max of sales: "
df.agg({'Sales':'max'}).show()

print  "same output as agg sales max right above"
group_data = df.groupBy("Company")
group_data.agg({"Sales": "max"}).show()

from  pyspark.sql.functions import countDistinct, avg, stddev
print "example of  using countDistinct"
df.select(countDistinct("Sales")).show()

print "example using average"
df.select(avg("Sales")).show()

print "giving an alias using same as above"
df.select(avg('Sales').alias('Average Sales')).show()

print "using standard diviation"
df.select(stddev('Sales')).show()

from pyspark.sql.functions import format_number

sales_std = df.select(stddev("Sales").alias('std'))
print "this is  the old: "
sales_std.show()

print "this is the new: "
sales_std.select(format_number('std',2)).show()

print "changing it again: "
sales_std.select(format_number('std',2).alias('std')).show()

# order and sorting things
print "showing the data frame again"
df.show()

print "order by examples: "
df.orderBy("Sales").show()

print "oder by  but decending: "
df.orderBy(df["Sales"].desc()).show()

