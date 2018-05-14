from pyspark.sql  import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import NGram

spark = SparkSession.builder.appName('nlp').getOrCreate()

sen_df = spark.createDataFrame([
    (0, 'Hi I heard about Spark'),
    (1, "I wish javacould use case classes"),
    (2, 'Logistic,regression,models,are,neat') 
], ['id','sentence'])

sen_df.show()

# toakenization = taking text and breaking it down into individual terms
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

regex_tokenizer = RegexTokenizer(inputCol='sentence', outputCol="words", pattern ='\\W')

count_tokens = udf(lambda words:len(words), IntegerType())

tokenized = tokenizer.transform(sen_df)
tokenized.show()

tokenized.withColumn('tokens', count_tokens(col('words'))).show()

rg_tokenized = regex_tokenizer.transform(sen_df)
rg_tokenized.withColumn('tokens', count_tokens(col('words'))).show()


# stop words remover (take out common  unimportant words)

sentence_data_frame = spark.createDataFrame([
    (0, ['I',  'saw',  'the', 'green', 'horse']),
    (1, ['Mary', 'had', 'a', 'little', 'lamb'])
], ['id', 'tokens'])

sentence_data_frame.show()

remover = StopWordsRemover(inputCol='tokens', outputCol="filtered") 
remover.transform(sentence_data_frame).show()

# n-gram find out the words that are asociated 
word_data_frame = spark.createDataFrame([
    (0, ["Hi", "I", "heard", "about", "Spark"]),
    (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
    (2, ["Logistic", "regression", "models", "are", "neat"])
], ["id",  "words"]) 

ngram = NGram(n=2, inputCol='words', outputCol='grams')
ngram.transform(word_data_frame).select('grams').show(truncate=False)


