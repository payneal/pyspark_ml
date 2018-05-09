# start a simple spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName(
    'data_frame_project').getOrCreate()

# load the walmart stock csvfile, have spark infer the data types
df = spark.read.csv(
    './data/walmart_stock.csv', header=True, inferSchema = True)

# what are the columns
print df.columns

# what does the Schema look like
df.printSchema()

# print out the first 5 columns
#for row in df.head(5):
    # print row
    # print "\n"

# describe() to learn about dataFrame
df.describe().show()

#  show up to two decimal places
df.describe().printSchema()

print "from here: "
from pyspark.sql.functions import format_number
result = df.describe()

def change_it(result, column, change_type, digit=None):
    if digit:
        return format_number(result[column].cast(change_type), digit).alias(column)
    return result[column].cast(change_type).alias(column)
    
result.select(
    result['summary'], 
    change_it(result, "Open", "float", 2),
    change_it(result, "High", "float", 2),
    change_it(result, "Low", "float", 2),
    change_it(result, "Close",'float', 2),
    change_it(result, "Volume", "int")).show()

# create new dataframe with a column called HV ratio that is the ratio of the high price versus volume of stock traded for a day

df2 = df.withColumn("HV Ratio", df['High']/df['Volume'])
df2.select('HV Ratio').show()

# what day had the peak high in price
print "this is the date: {}".format(df.orderBy(df['High'].desc()).head(1)[0][0])

# what is the mean of the close column
from pyspark.sql.functions import mean
df.select(mean("Close")).show()

# what is  the max and min of the volume column
from pyspark.sql.functions import max, min
df.select(max('Volume'), min("Volume")).show()

# how many days was the closelower yhan 60  dollars (3 ways)
#
# print "option one: "
# print df.filter('Close < 60').count()
# print "option two(more pythonic): "
# print df.filter(df["Close"] < 60).count()
# print in a data frame
from pyspark.sql.functions import count
result = df.filter(df['Close'] < 60 )
result.select(count('Close')).show()

# what precentage of the time was the high greater than 80 $
print "precent: {}".format(df.filter(df['high']>80).count() *1.0 / df.count()* 100)

# what is the pearson corrrelation between high and volume
from pyspark.sql.functions import corr
df.select(corr('High', 'Volume')).show()

# what is the max high per year
from pyspark.sql.functions import year
yeardf = df.withColumn("Year", year(df["Date"]))
max_df = yeardf. groupBy('Year').max()
max_df.select("Year", "max(High)").show()

# what is the averageclose for each calendar month
from pyspark.sql.functions import month
monthdf = df.withColumn('Month',month('Date'))
monthavg = monthdf.select(['Month', 'Close']).groupBy('Month').mean()
monthavg.select('Month', 'avg(Close)').orderBy('Month').show()
