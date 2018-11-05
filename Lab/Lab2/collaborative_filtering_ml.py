from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct
import pandas as pd
import os

spark = SparkSession.builder.master("local").appName("Collborative Filering").getOrCreate()
training = spark.read.csv('training', inferSchema =True).toDF('itemId', 'rating', 'userId')
test = spark.read.csv('test', inferSchema =True).toDF('itemId', 'rating', 'userId')

# training = training.withColumn("itemId", training["itemId"].cast(IntegerType()))\
#     .withColumn('userId', training['userId'].cast(IntegerType())).withColumn('rating', training['rating'].cast(FloatType()))
# test = test.withColumn("itemId", test["itemId"].cast(IntegerType()))\
#     .withColumn('userId', test['userId'].cast(IntegerType())).withColumn('rating', test['rating'].cast(FloatType()))

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
K = 5
train = False
if train:
    if not os.path.isdir('als-model'):
        als = ALS(maxIter=20, regParam=1, rank=50, userCol="userId", itemCol="itemId", ratingCol="rating",
                  coldStartStrategy="drop")
        model = als.fit(training)
        model.save('als-model')
    else:
        # load model
        print('loading model')
        model = ALSModel.load('als-model')


    test_users = test.select('userId').distinct()
    test_users.show()
    test_recs = model.recommendForUserSubset(test_users, K).select("userId", "recommendations.itemId")
    test_recs.show()

    recs_list = test_recs.rdd.collect()

    pk.dump(recs_list, open('recs.p', 'wb'))

print('loading predictions')
recs_list = pk.load(open('recs.p', 'rb'))
def is_converted(row, k):
    userId = row.userId
    recs = row.itemId[:k]
    actual = test.where(test.userId == userId).select('itemId').rdd.map(lambda row: row.itemId).collect()
    training_actual = training.where(training.userId == userId).select('itemId').rdd.map(lambda row: row.itemId).collectAsSet
    # print(actual)
    # print(recs)
    for rec in recs:
        if rec in training_actual:
            print('problem!')
        if rec in actual:
            print('got one')
            return 1
    return 0


for i in range(1, K+1):
    print('predicting conversion rate @K={0}'.format(i))
    count = 0
    for row in recs_list:
        if is_converted(row, i):
            count += 1
    rate = count/len(recs_list)
    print('K={0}, conversion rate={1}'.format(i, rate))



