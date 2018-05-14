from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from  pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName('rec').getOrCreate()

data = spark.read.csv('./data/movielens_ratings.csv', inferSchema=True, header= True)

data.show()

data.describe().show()

#  split data train/test but keep in mind that its hard to validate a recomendation system when info is subjective
training, test = data.randomSplit([0.8, 0.2]) 

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol='movieId', ratingCol='rating')
model = als.fit(training)

predictions = model.transform(test)

predictions.show()

evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

rmse = evaluator.evaluate(predictions)

print 'RMSE'
print rmse

single_user  = test.filter(test['userId'] == 11) .select(['movieId', 'userId'])

single_user.show()

recommendations = model. transform(single_user)

recommendations.orderBy('prediction', ascending=False).show()

#  cold start issue
# * appears when new user comes to system with no previous data
# * can have user list and rank some movies in system
# * can have user associated with another user
# * this is a common issue with reccomendation systems

