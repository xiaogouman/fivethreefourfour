from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct

spark = SparkSession.builder.master("local").appName("Classify Urls").getOrCreate()

rawdf = spark.read.csv("ratings_Digital_Music.csv").toDF('userId','itemId','rating','timestamp')
rawdf.createOrReplaceTempView("useritem")
# row counts = 836006
spark.sql('SELECT COUNT(*) FROM useritem')
# item counts = 266414
spark.sql('SELECT COUNT(DISTINCT itemId) FROM useritem')
# user counts = 478235
spark.sql('SELECT COUNT(DISTINCT userId) FROM useritem')
# number of users who only rate for 1 item: 358615
spark.sql('SELECT COUNT(*) FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count == 1)')
# number of items who got rated by 1 user: 186753
spark.sql('SELECT COUNT(*) FROM (SELECT DISTINCT COUNT(*) AS count, itemId FROM useritem GROUP BY itemId HAVING count == 1)')


# select user who rate for > 1 items and items rated by > 1 user
users = 'SELECT userId FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count > 1)'
items = 'SELECT itemId FROM (SELECT DISTINCT COUNT(*) AS count, itemId FROM useritem GROUP BY itemId HAVING count > 1)'
# after filter out the sparse part: 353553
df = spark.sql('SELECT * FROM useritem WHERE userId IN ('+ users +') AND itemId IN (' + items + ')')
print ('count of filtered db: {0}'.format(df.count()))

userIdIntMap = df.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
itemIdIntMap = df.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()


ratings = df.rdd.map(lambda d:Rating(userIdIntMap.get(d.userId), itemIdIntMap.get(d.itemId), float(d.rating)))
(training, test) = ratings.randomSplit([0.8, 0.2])
rank = 50
numIterations = 10
l = 0.001
ranks = [10, 50]
ls = [1, 0.1, 0.01, 0.001, 0.0001]

for rank in ranks:
    for l in ls:
        model = ALS.train(training, rank, numIterations, l)
        print ("model with rank = {0}, l = {1}".format(rank, l))

        # Evaluate the model on training data
        train_data = training.map(lambda p: (p[0], p[1]))
        train_pred = model.predictAll(train_data).map(lambda r: ((r[0], r[1]), r[2]))
        train_ratesAndPreds = training.map(lambda r: ((r[0], r[1]), r[2])).join(train_pred)
        train_MSE = train_ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        print("Train Mean Squared Error = " + str(train_MSE))

        # Evaluate the model on test data
        test_data = test.map(lambda p: (p[0], p[1]))
        test_pred = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
        test_ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(test_pred)
        test_MSE = test_ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        print("Test Mean Squared Error = " + str(test_MSE))

# convesion rate
#K = 10
#for k in range(K):
#	model.predictAll()

#https://colobu.com/2015/11/30/movie-recommendation-for-douban-users-by-spark-mllib/

'''
# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=50, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
als.checkpointInterval = 2
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
'''
