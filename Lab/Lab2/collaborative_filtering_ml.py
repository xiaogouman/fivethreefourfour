from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct

spark = SparkSession.builder.master("local").appName("Classify Urls").getOrCreate()

rawdf = spark.read.csv("/home/zuo/Documents/Masters/CS5344/Lab/Lab2/ratings_Digital_Music.csv").toDF('userId','itemId','rating','timestamp')
rawdf.createOrReplaceTempView("useritem")
# row counts = 836006
spark.sql('SELECT COUNT(*) FROM useritem')
# item counts = 266414
spark.sql('SELECT COUNT(DISTINCT itemId) FROM useritem')
# user counts = 478235
spark.sql('SELECT COUNT(DISTINCT userId) FROM useritem')
# number of users who only rate for 1 item: 358615
spark.sql('SELECT COUNT(*) FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count == 1)')
# select user who rate for > 1 items and items rated by > 1 user
users = 'SELECT userId FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count > 1)'
# after filter out the sparse part: 353553
df = spark.sql('SELECT * FROM useritem WHERE userId IN ('+ users +') AND itemId IN (' + items + ')')
print ('count of filtered db: {0}'.format(df.count()))

userIdIntMap = df.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
itemIdIntMap = df.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()

ratings = df.rdd.map(lambda d:Row(userId=userIdIntMap.get(d.userId), itemId=itemIdIntMap.get(d.itemId), rating=float(d.rating))).toDF()
(training, test) = ratings.randomSplit([0.8, 0.2])
rank = 50
numIterations = 10
l = 0.001
ranks = [10, 50]
ls = [1, 0.1, 0.01, 0.001, 0.0001]

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=10, regParam=0.001, rank=50, userCol="userId", itemCol="itemId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the MSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",
                                predictionCol="prediction")
mse = evaluator.evaluate(predictions)
print("Mean-square error = " + str(mse))

K = 10
k = 3
# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(k).show()
userRecs.rdd.map(lambda r: row.userId)


def conversion(userId, recommendations):
	

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)

