from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct
import pandas as pd
import os

spark = SparkSession.builder.master("local").appName("Classify Urls").getOrCreate()
columns = ['userId','itemId','rating','timestamp']
if True:#not os.path.isfile('filtered_data.csv'):
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
    # select user who rate for > 1 items and items rated by > 1 user
    users = 'SELECT userId FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count > 1)'
    items = 'SELECT itemId FROM (SELECT DISTINCT COUNT(*) AS count, itemId FROM useritem GROUP BY itemId HAVING count > 1)'

    # after filter out the sparse part: 353553
    df = spark.sql('SELECT * FROM useritem WHERE userId IN ('+ users +')')#AND itemId IN (' + items + ')')

    # df_pandas = pd.DataFrame(df, columns=columns)
    # df_pandas.to_csv("filtered_data.csv", header=True)
else:
    print('loading processed data')
    df = spark.read.csv("filtered_data.csv", header=True).toDF()

print('count of filtered db: {0}'.format(df.count()))

userIdIntMap = df.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
itemIdIntMap = df.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()

ratings = df.rdd.map(lambda d:Row(userId=userIdIntMap.get(d.userId), itemId=itemIdIntMap.get(d.itemId), rating=float(d.rating))).toDF()
(training, test) = ratings.randomSplit([0.8, 0.2])


############## test on the paramters for ALS model
numIterations = 20
ranks = [50]
ls = [1, 0.1, 0.01, 0.001, 0.0001]

test_model_params = False
if test_model_params:
    for rank in ranks:
        for l in ls:
            print('rank={}, l={}'.format(rank, l))

            # Build the recommendation model using ALS on the training data
            # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
            als = ALS(maxIter=numIterations, regParam=l, rank=rank, userCol="userId", itemCol="itemId", ratingCol="rating",
                      coldStartStrategy="drop")
            model = als.fit(training)

            # Evaluate the model by computing the MSE on the test data
            predictions = model.transform(test)
            evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",
                                            predictionCol="prediction")
            mse = evaluator.evaluate(predictions)
            print("test Mean-square error = " + str(mse))

            predictions_train = model.transform(training)
            evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",
                                            predictionCol="prediction")
            mse = evaluator.evaluate(predictions_train)
            print("train Mean-square error = " + str(mse))


############### calculate conversion rate ##################
K = 5

als = ALS(maxIter=20, regParam=1, rank=50, userCol="userId", itemCol="itemId", ratingCol="rating",
          coldStartStrategy="drop")

if not os.path.isdir('als-model'):
    model = als.fit(training)
    model.save('als-model')
else:
    # load model
    print('loading model')
    model = ALSModel.load('als-model')

test_users = test.select('userId').distinct()
test_recs = model.recommendForUserSubset(test_users, K)

def is_converted(row, k):
    userId = row.userId
    recs = row.item[:k]
    actual = test.select('userId == ' + userId).rdd.map(lambda row: row.itemId).collect()
    # for rec in recs:
    #     if rec in actual


test_recs.show()
for i in range(1, K+1):
    result = test_recs.rdd.map(lambda row: is_converted(row)).collect()
    rate = sum(result)/len(result)
    print('K={0}, conversion rate={1}'.format(i, rate))



