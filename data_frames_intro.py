from pyspark.sql import SparkSession
from pyspark.sql.types import(StructField, StringType, IntegerType, StructType)

# into spark data frame basics

spark = SparkSession.builder.appName("Basics").getOrCreate()
df = spark.read.json('./data/people.json')

print "this is show:  "
df.show()

print "this is print schema"
df.printSchema()

print "this is colums"
df.columns

print "this is describe"
df.describe().show()

data_schema = [
    StructField('age',IntegerType(), True),
    StructField('name',StringType(), True)]

final_struc = StructType(fields=data_schema)

df = spark.read.json('./data/people.json', schema=final_struc)

print "this is th schema: "
df.printSchema()

print "this is the show"
df.show()

print "show me the type: "
print type(df['age'])


print "get me the first row: "
print df.head(1)

print "get me the first 2 rows: "
print df.head(2)

print "get me first two rows but show me 2nd entry"
print df.head(2)[1]
print "prove that this is a row"
print type(df.head(2)[1])


print "select list of columns (data frame)"
df.select(['age', 'name']).show()

print "adding in a col or replacing an exisiting col"
df.withColumn('new_age', df['age']).show()

print "adding column with new column manipulation"

print "here is a regualr example"
df.withColumn('double_age', df['age']/2).show()

print "here is a function example"
def change_it(umm):
   umm =  umm *2
   umm += 15
   return umm
df.withColumn('double_age', change_it(df['age'])).show()

print  "show that the changes are  not perm"
df.show()

print "here is how you rename a coloum"
df.withColumnRenamed('age','my_new_age').show()

print "how to do direc queries: "
df.createOrReplaceTempView('people')

results = spark.sql("SELECT * FROM people where age=30")
results.show()
