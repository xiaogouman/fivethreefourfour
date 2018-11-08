from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.recommendation import Rating
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.functions import countDistinct
import pandas as pd
import os
import random

spark = SparkSession.builder.master("local").appName("Collborative Filering").getOrCreate()

# create df
rawdf = spark.read.csv("ratings_Musical_Instruments.csv").toDF('userId', 'itemId', 'rating', 'timestamp')
rawdf.createOrReplaceTempView("useritem")

# row counts = 836006
count = spark.sql('SELECT COUNT(*) FROM useritem')
count.show()

# item counts = 266414
itemId_count = spark.sql('SELECT COUNT(DISTINCT itemId) FROM useritem')
itemId_count.show()

# user counts = 478235
userId_count = spark.sql('SELECT COUNT(DISTINCT userId) FROM useritem')
userId_count.show()

# number of users who only rate for 1 item: 358615
userId_one_item = spark.sql(
    'SELECT COUNT(*) FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count == 1)')
userId_one_item.show()

# select user who rate for > 1 items and items rated by > 1 user
users = 'SELECT userId FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count > 2)'

# after filter out the sparse part: 353553
df = spark.sql('SELECT * FROM useritem WHERE userId IN (' + users + ')')
print('count of filtered db: {0}'.format(df.count()))

userIdIntMap = df.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
itemIdIntMap = df.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()

ratings = df.rdd.map(lambda d:Row(userIdIntMap.get(d.userId), itemIdIntMap.get(d.itemId), float(d.rating), int(d.timestamp))).toDF()
ratings = ratings.toDF('userId','itemId','rating','timestamp')

grouped = ratings.orderBy(F.desc('timestamp')).groupBy(ratings.userId).agg(F.collect_list(ratings.itemId).alias("itemId"), F.collect_list(ratings.rating).alias('rating'))
grouped.show()
test_list = []
for user_data in grouped.rdd.collect():
    userId = user_data.userId
    item = user_data.itemId
    rating = user_data.rating
    length = int(len(item)/10)
    for i in range(length):
        test_list.append(Row(userId, item[i], rating[i]))

# test = spark.createDataFrame(test_list).toDF('userId','itemId','rating')
# training = ratings.drop('timestamp').subtract(test)
# print(test.count())
# print(training.count())

(training, test) = ratings.drop('timestamp').randomSplit([0.9, 0.1])
training.toPandas().to_csv('train.csv', index=False)
test.toPandas().to_csv('test.csv', index=False)
