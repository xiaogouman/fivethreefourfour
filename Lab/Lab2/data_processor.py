from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.recommendation import Rating
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct
import pandas as pd
import os

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
users = 'SELECT userId FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count > 1)'
items = 'SELECT itemId FROM (SELECT DISTINCT COUNT(*) AS count, itemId FROM useritem GROUP BY itemId HAVING count > 1)'

# after filter out the sparse part: 353553
df = spark.sql('SELECT * FROM useritem WHERE userId IN (' + users + ')')  # AND itemId IN (' + items + ')')
print('count of filtered db: {0}'.format(df.count()))

userIdIntMap = df.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
itemIdIntMap = df.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()

ratings = df.rdd.map(lambda d:Row(userIdIntMap.get(d.userId), itemIdIntMap.get(d.itemId), float(d.rating))).toDF()
ratings = ratings.toDF('userId','itemId','rating')
ratings.show()
(training, test) = ratings.randomSplit([0.8, 0.2])
training.toPandas().to_csv('train.csv', index=False)
test.toPandas().to_csv('test.csv', index=False)
