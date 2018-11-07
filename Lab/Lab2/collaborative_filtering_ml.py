from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from collections import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel


spark = SparkSession.builder.appName("Collborative Filering")\
    .master('local[3]')\
    .config('spark.executor.heartbeatInterval','20s')\
    .config('spark.executor.memory', '3g') \
    .config('spark.driver.memory', '20g') \
    .getOrCreate()
sc = spark.sparkContext
training = spark.read.csv('train.csv', inferSchema =True, header=True).toDF('userId','itemId', 'rating')
test = spark.read.csv('test.csv', inferSchema =True, header=True).toDF('userId','itemId', 'rating')

############## test on the paramters for ALS model ############
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


############### traing model ##################

test_users = test.select('userId').distinct()
test_user_set = set(test_users.rdd.map(lambda r: r.userId).collect())
K = 5

als = ALS(maxIter=20, rank=50, regParam=1, userCol="userId", itemCol="itemId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

############## conversion rate ##############
test_recs = model.recommendForUserSubset(test_users, 250).select("userId", "recommendations.itemId")
test_recs_dict = defaultdict(list)
for row in test_recs.rdd.collect():
    test_recs_dict[row.userId] = list(row.itemId)

train_rated = training.where(F.col('userId').isin(test_user_set)).groupBy(training.userId).agg(F.collect_set(training.itemId).alias("itemId"))
train_dict = defaultdict(set)
for row in train_rated.rdd.collect():
    train_dict[row.userId] = set(row.itemId)

test_rated = test.groupBy(test.userId).agg(F.collect_set(test.itemId).alias("itemId"))
test_dict = defaultdict(set)
for row in test_rated.rdd.collect():
    test_dict[row.userId] = set(row.itemId)

print('collect finish')
test_user_count = test_users.count()
for i in range(1, K+1):
    count = 0
    for userId in test_user_set:
        recs = test_recs_dict[userId]
        train = train_dict[userId]
        test = test_dict[userId]
        actual_recs = set([x for x in recs if x not in train][:i])
        # if set(recs).intersection(train):
        #     print('problem')
        if actual_recs.intersection(test):
            count += 1
    rate = count/test_user_count
    print('K={0}, conversion rate={1}'.format(i, rate))




