from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('miss').getOrCreate()
df = spark.read.csv(
    './data/ContainsNull.csv', header=True, inferSchema = True)

print "show the data"
df.show()

print "drop any row with missing data"
df.na.drop().show()

print "drop rows that are missing data with a threshold: "
#  row must have 2 nulls in data to be dropped
df.na.drop(thresh=2).show()

print "only drop the row if all the columns are null"
df.na.drop(how="all").show()

print "i null is in certain row drop it all"
df.na.drop(subset=['Sales']).show()

# fill in missing values
print "showing the schema"
df.printSchema()

print "fill in missing value"
df.na.fill("FILL VALUE").show()

print "fill in using a number and because sales is double"
#knows its a number so it handles slaes data column 
df.na.fill(0).show()

print  "fill in a a certain column"
df.na.fill("No Name", subset=['Name']).show()

# filling in data with average foir missing info
from pyspark.sql.functions import mean
print "find sales mean value"
mean_val = df.select(mean(df['Sales'])).collect()
mean_sales = mean_val[0][0]
print "here is mean valuea: {}".format(mean_sales)
df.na.fill(mean_sales, ['Sales']).show() 

# same thing but one line
print "same thing but its a one linner"
df.na.fill(df.select(mean(df["Sales"])).collect()[0][0], ['Sales']).show()
