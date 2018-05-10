from pyspark.sql import SparkSession
from pyspark.ml.feature import(VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName(
    'titantic').getOrCreate()
df = spark.read.csv("./data/titanic.csv", inferSchema=True, header=True)
df.printSchema()

print "these are columns {}".format(df.columns)

my_cols = df.select([
    'Survived', 'Pclass', 'Sex',
    'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# drop all missing data
my_final_data = my_cols.na.drop()

gender_indexer =StringIndexer(inputCol='Sex', outputCol='SexIndex')

gender_encoder = OneHotEncoder(inputCol="SexIndex", outputCol='SexVec')


embark_indexer = StringIndexer(inputCol='Embarked', outputCol='EmbarkedIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkedIndex', outputCol='EmbarkVec')

assembler = VectorAssembler(inputCols=['Pclass', 'SexVec', 'EmbarkVec',  'Age', 'SibSp', 'Parch', 'Fare'], outputCol='features')

log_reg_titantic = LogisticRegression(featuresCol='features', labelCol='Survived')

pipeline = Pipeline(stages=[gender_indexer, embark_indexer, gender_encoder, embark_encoder, assembler, log_reg_titantic])

train_data, test_data = my_final_data.randomSplit([0.7, 0.3])

fit_model = pipeline.fit(train_data)

results = fit_model.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Survived')

# results.select('Survived', 'prediction').show()

AUC = my_eval.evaluate(results)

print "this is AUC: {}".format(AUC)
