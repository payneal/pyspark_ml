from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF,IDF, Tokenizer
from pyspark.ml.feature import CountVectorizer

spark = SparkSession.builder.appName('nlp').getOrCreate()

sentence_data = spark.createDataFrame([
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
], ["label", "sentence"])

sentence_data.show()

tokenizer = Tokenizer(inputCol='sentence', outputCol="words")
words_data = tokenizer.transform(sentence_data)

words_data.show()

hashing_tf = HashingTF(inputCol='words', outputCol="rawFeatures") 

featurized_data = hashing_tf.transform(words_data)

idf= IDF (inputCol="rawFeatures", outputCol="features")

idf_model = idf.fit(featurized_data)

rescaled_data = idf_model.transform(featurized_data)

rescaled_data.select('label', 'features').show(truncate=False)

df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" ")) 
], ["id", "words"])

cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)

model = cv.fit(df)

result = model.transform(df)

result.show(truncate=False)
