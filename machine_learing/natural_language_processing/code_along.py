from pyspark.sql  import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import(Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName('nlp').getOrCreate()
data = spark.read.csv('./data/SMSSpamCollection.csv', inferSchema=True, sep='\t')
data.show()


data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')
data.show()

data = data.withColumn('length', length(data['text']))
data.show()

data.groupBy('class').mean().show()

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stop_remove = StopWordsRemover(inputCol="token_text", outputCol='stop_token')
count_vrc = CountVectorizer(inputCol='stop_token', outputCol="c_vec")
idf = IDF(inputCol='c_vec', outputCol='tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol="class", outputCol='label')


clean_up = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')

nb = NaiveBayes()

data_prep_pipe = Pipeline(stages=[ham_spam_to_numeric, tokenizer, stop_remove, count_vrc, idf, clean_up])

cleaner = data_prep_pipe.fit(data)

clean_data  = cleaner.transform(data)

clean_data = clean_data.select('label', 'features')

clean_data.show()


# spliting up to test and train
training, test = clean_data.randomSplit([0.7, 0.3])

spam_detector = nb.fit(training)
data.printSchema()

test_results = spam_detector.transform(test)
test_results.show()

acc_eval = MulticlassClassificationEvaluator()

acc = acc_eval.evaluate(test_results)

print "ACC of NB Model"
print acc
