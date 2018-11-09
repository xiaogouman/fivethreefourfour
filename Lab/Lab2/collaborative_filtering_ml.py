from pyspark.sql import SparkSession
from pyspark.sql import Row
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

# read dataset
all = spark.read.csv('ratings_Musical_Instruments.csv').toDF('userId','itemId', 'rating','timestamp')
training = spark.read.csv('train.csv', inferSchema =True, header=True).toDF('userId','itemId','rating')
test = spark.read.csv('test.csv', inferSchema =True, header=True).toDF('userId','itemId','rating')

# map String itemId and userId to integers
userIdIntMap = all.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
itemIdIntMap = all.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()

training = training.rdd.map(lambda d: Row(userIdIntMap.get(d.userId), itemIdIntMap.get(d.itemId), float(d.rating))).toDF()
training = training.toDF('userId','itemId', 'rating')
test = test.rdd.map(lambda d: Row(userIdIntMap.get(d.userId), itemIdIntMap.get(d.itemId), float(d.rating))).toDF()
test = test.toDF('userId','itemId', 'rating')
print('test: ', test.count(), 'train: ',training.count())

############### traing model ##################

test_users = test.select('userId').distinct()
test_user_set = set(test_users.rdd.map(lambda r: r.userId).collect())
K = 5

als = ALS(maxIter=20, rank=50, regParam=1, userCol="userId", itemCol="itemId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)
print('training finish')

############## conversion rate ##############
# recommend 270 items for each user and remove those ones in training set
test_recs = model.recommendForUserSubset(test_users, 270).select("userId", "recommendations.itemId")
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

# calculate conversion rate
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
