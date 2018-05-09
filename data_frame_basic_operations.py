from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ops').getOrCreate()
df = spark.read.csv('./data/appl_stock.csv', inferSchema=True, header=True)
# df.printSchema()
# df.show()
# print  df.head(3)[0]

# grab all data with closin price < 500 but only show open & close column
df.filter("close < 500").select(["Open", "Close"]).show()

# another example
df.filter(df['Close'] < 500).select('Volume').show()

# filtering multiple condidtions
df.filter( (df['Close'] < 200) &  (df['Open']  > 200)).show()

# example of using not
df.filter((df["Close"] < 200 ) & ~(df['Open'] > 200) ).show()

# another example of filter
df.filter(df['Low'] == 197.16).show()

# example of collecting a search
result = df.filter(df['Low']  == 197.16).collect()

# this is  an example of what collection does
print result

# exampole of grabing inf from collection
print "#1 item in the collection "
print result[0]

print "second go around"

# trying again
print result[0]["Open"]

# turning a row into a dictionary
row = result[0]
print "this is the dict form for a row: {}".format(row.asDict())
print "this is the dict form for a row: {}".format(row.asDict()["Open"])
