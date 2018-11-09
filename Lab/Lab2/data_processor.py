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
import sys

spark = SparkSession.builder.master("local").appName("Collborative Filering").getOrCreate()

# create df
rawdf = spark.read.csv("ratings_Musical_Instruments.csv").toDF('userId', 'itemId', 'rating', 'timestamp')
rawdf.createOrReplaceTempView("useritem")

# row counts = 500176
spark.sql('SELECT COUNT(*) as total_count FROM useritem').show()

# item counts = 83046
spark.sql('SELECT COUNT(DISTINCT itemId) as item_count FROM useritem').show()

# user counts = 339231
spark.sql('SELECT COUNT(DISTINCT userId) as user_count FROM useritem').show()

# number of users who only rate for 1 item: 270914
spark.sql(
    'SELECT COUNT(*) as user_one_item_count FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count == 1)').show()

# select user who rate for > 2 items
ratings = spark.sql('SELECT * FROM useritem WHERE userId IN (SELECT userId FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count > 2)').show()

# after filter out the sparse part: 150708
print('count of filtered db: {0}'.format(ratings.count()))


if sys.argv[0] == 'top10':
    # generate test data by selecting the top 10 ratings for each user(if he has equal or more than 10 ratings)
    print('top10')
    grouped = ratings.orderBy(F.desc('timestamp')).groupBy(ratings.userId).agg(F.collect_list(ratings.itemId).alias("itemId"), F.collect_list(ratings.rating).alias('rating'))
    test_list = []
    for user_data in grouped.rdd.collect():
        userId = user_data.userId
        item = user_data.itemId
        rating = user_data.rating
        length = int(len(item)/10)
        for i in range(length):
            test_list.append(Row(userId, item[i], rating[i]))

    test = spark.createDataFrame(test_list).toDF('userId', 'itemId', 'rating')
    training = ratings.drop('timestamp').subtract(test)
else:
    # random split data set
    print('random')
    (training, test) = ratings.drop('timestamp').randomSplit([0.9, 0.1])

training.toPandas().to_csv('train.csv', index=False)
test.toPandas().to_csv('test.csv', index=False)
print(test.count(), training.count())

