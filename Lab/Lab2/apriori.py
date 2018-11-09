from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.window import Window as W
from collections import defaultdict

spark = SparkSession.builder.appName("Apriori")\
    .master('local[3]')\
    .config('spark.executor.heartbeatInterval','20s')\
    .config('spark.executor.memory', '3g') \
    .config('spark.driver.memory', '20g') \
    .getOrCreate()

# The dataset we have used is musical instrument
df = spark.read.json("meta_Musical_Instruments.json")
df.show()

# We have observed for some metadata ths asin, related and bought_together fields are Null. We have filtered out those fields.
df = df.filter(df.asin.isNotNull())
df = df.filter(df.related.isNotNull())
df = df.filter(df.related.also_viewed.isNotNull())

# We have set the mimimum number of bought_together items to be 2. So in total, there will be at least 3 items in a row.
rawData= df.rdd.filter(lambda row: len(row.related.also_viewed) >= 1)

trainData = rawData

# We have reconstruced the train data frame work. Each reconstructed row will contain asin item and item which a user bought together.
trainDF = trainData.map(lambda row: Row(bought=list(set([row.asin] + row.related.also_viewed )))).toDF()

# We have used FPGrowth algorithm.
fpGrowth = FPGrowth(itemsCol="bought", minSupport=0.001, minConfidence=0.1)
model = fpGrowth.fit(trainDF)
print('train finish')
print(model.associationRules.count())
model.associationRules.show()



# conversion rate
test = spark.read.csv('test.csv', inferSchema =True, header=True).toDF('userId','itemId','rating')

# test user set
test_users = test.select('userId').distinct()
test_user_set = set(test_users.rdd.map(lambda r: r.userId).collect())
print('test user', test_users.count())

# actual bought in test set
test_rated = test.groupBy(test.userId).agg(F.collect_set(test.itemId).alias("itemId"))

test_dict = defaultdict(list)
for row in test_rated.rdd.collect():
    test_dict[row.userId] = list(row.itemId)

test_bought = test_rated.toDF('userId', 'bought')

predictions = model.transform(test_bought).select(['userId', 'prediction'])
print('finish transform')
print('predictions', predictions.count())

test_preds = defaultdict(list)
for row in predictions.rdd.collect():
    test_preds[row.userId] = list(row.prediction)
print(test_preds)
K = 10
test_user_count = test_users.count()
for i in range(1, K+1):
    count = 0
    for userId in test_user_set:
        # pred = set(predictions.select(predictions.userId == userId).prediction[:i])
        # print(pred)
        pred = set(test_preds[userId][:i])
        test = set(test_dict[userId])
        print('pred:', pred, 'actual:', test)
        if pred.intersection(test):
            count += 1
    rate = count/test_user_count
    print('K={0}, conversion rate={1}'.format(i, rate))


