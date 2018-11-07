from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct

#.config('spark.driver.memory', '20g')\
spark = SparkSession.builder.appName("Collborative Filering")\
    .master('local[3]')\
    .config('spark.executor.heartbeatInterval','20s')\
    .config('spark.executor.memory', '3g') \
    .config('spark.driver.memory', '20g') \
    .getOrCreate()
sc = spark.sparkContext


training = spark.read.csv('train.csv', inferSchema =True, header=True).toDF('userId','itemId','rating')
test = spark.read.csv('test.csv', inferSchema =True, header=True).toDF('userId','itemId','rating').rdd
training_rdd = training.rdd
test_user_set = set(test.map(lambda r: r.userId).distinct().collect())
print(test_user_set)

train = False
if train:
    model = ALS.train(training_rdd, rank=50, iterations=20)#, rank=50, iterations=20, lambda_= 1)

    print('recommend start')
    recs_all = model.recommendProductsForUsers(180)
    # (54175, (Rating(user=54175, product=37684, rating=4.999785805913131), Rating(user=54175, product=26802, rating=4.994706715965597)))
    print('recommend finish')
    recs_all = recs_all.filter(lambda r: r[0] in test_user_set).saveAsPickleFile('recs')
recs_all = sc.pickleFile('recs')
train_rated = training.where(F.col('userId').isin(test_user_set)).groupBy(training.userId).agg(F.collect_set(training.itemId))
train_rated.show()

K = 10
for i in range(1, K+1):
    count = 0
    for userId in test_user_set:
        train_rated = set(training.filter(lambda r: r.userId == userId).map(lambda r: r.itemId).collect())


        recs = recs_all.filter(lambda r: r[0] == userId).flatMap(lambda r: [rating.product for rating in r[1]])
        # print('recs: ', recs.collect())
        # print('train_rated', train_rated)
        recs = set(recs.filter(lambda item: item not in train_rated).take(i))
        print('userId: ', userId, 'recs: ', recs)
        joint_count = test.filter(lambda r: r.userId == userId and r.itemId in recs).count()
        if joint_count > 0:
            print('got one')
            count += 1
    print('conversion rate @K={0}: {1}'.format(i, count/len(test_user_set)))




# rank = 50
# numIterations = 10
# l = 0.001
# ranks = [10, 50]
# ls = [1, 0.1, 0.01, 0.001, 0.0001]
#
# for rank in ranks:
#     for l in ls:
#         model = ALS.train(training, rank, numIterations, l)
#         print ("model with rank = {0}, l = {1}".format(rank, l))
#
#         # Evaluate the model on training data
#         train_data = training.map(lambda p: (p[0], p[1]))
#         train_pred = model.predictAll(train_data).map(lambda r: ((r[0], r[1]), r[2]))
#         train_ratesAndPreds = training.map(lambda r: ((r[0], r[1]), r[2])).join(train_pred)
#         train_MSE = train_ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
#         print("Train Mean Squared Error = " + str(train_MSE))
#
#         # Evaluate the model on test data
#         test_data = test.map(lambda p: (p[0], p[1]))
#         test_pred = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
#         test_ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(test_pred)
#         test_MSE = test_ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
#         print("Test Mean Squared Error = " + str(test_MSE))

# convesion rate
#K = 10
#for k in range(K):
#	model.predictAll()

#https://colobu.com/2015/11/30/movie-recommendation-for-douban-users-by-spark-mllib/


