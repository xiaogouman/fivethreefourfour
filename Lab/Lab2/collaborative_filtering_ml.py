from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct
import pandas as pd
import os

spark = SparkSession.builder.master("local[4]")\
    .appName("Collborative Filering")\
    .config("spark.driver.memory","20g")\
    .getOrCreate()
sc = spark.sparkContext
training = spark.read.csv('train.csv', inferSchema =True, header=True).toDF('userId','itemId', 'rating')
test = spark.read.csv('test.csv', inferSchema =True, header=True).toDF('userId','itemId', 'rating')

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
import pickle as pk

test_users = test.select('userId').distinct()
K = 10
train = True
if train:
    als = ALS(maxIter=20, rank=50, userCol="userId", itemCol="itemId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    # test_users.show()
    test_recs = model.recommendForUserSubset(test_users, 180).select("userId", "recommendations.itemId")
    # test_recs.show()

#     print('saving predictions')
#     test_recs.rdd.saveAsPickleFile('recs-ml')
#
# print('loading predictions')
# recs_rdd = sc.pickleFile('recs-ml')
# print(recs_rdd.take(2))
def is_converted(row, k):
    userId = row.userId
    recs = row.itemId
    actual = set(test.where(test.userId == userId).select('itemId').rdd.map(lambda row: row.itemId).collect())
    training_actual = set(training.where(training.userId == userId).select('itemId').rdd.map(lambda row: row.itemId).collect())
    recs_exclude_training = [x for x in recs if x not in training_actual][:k]
    # print(actual)
    # print(recs)
    for rec in recs_exclude_training:
        # if rec in training_actual:
        #     print('problem!')
        if rec in actual:
            print('got one')
            return True
    return False

test_user_count = test_users.count()
for i in range(1, K+1):
    print('predicting conversion rate @K={0}'.format(i))
    count = 0
    for row in test_recs.rdd.collect():
        if is_converted(row, i):
            count += 1
    rate = count/test_user_count
    print('K={0}, conversion rate={1}'.format(i, rate))
    # count = 0
    # for userId in test_users.rdd.collect():
    #     train_rated = set(training.filter(lambda r: r.userId == userId).map(lambda r: r.itemId).collect())
    #
    #     recs = recs_all.filter(lambda r: r[0] == userId).flatMap(lambda r: [rating.product for rating in r[1]])
    #     # print('recs: ', recs.collect())
    #     # print('train_rated', train_rated)
    #     recs = set(recs.filter(lambda item: item not in train_rated).take(i))
    #     print('userId: ', userId, 'recs: ', recs)
    #     joint_count = test.filter(lambda r: r.userId == userId and r.itemId in recs).count()
    #     if joint_count > 0:
    #         print('got one')
    #         count += 1
    # print('conversion rate @K={0}: {1}'.format(i, count/test_users.count()))



