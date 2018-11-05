from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct
import pandas as pd
import os

spark = SparkSession.builder.master("local").appName("Collborative Filering").getOrCreate()

if os.path.isdir('training') and os.path.isdir('test'):
    training = spark.read.csv('training').toDF('itemId', 'rating', 'userId')
    test = spark.read.csv('test').toDF('itemId', 'rating', 'userId')
    print('training size: {0}'.format(training.count()))
    print('test size: {0}'.format(test.count()))
    training.show()
    test.show()
else:
    if not os.path.isdir('filtered_data'):
        rawdf = spark.read.csv("ratings_Digital_Music.csv").toDF('userId', 'itemId', 'rating', 'timestamp')
        rawdf.createOrReplaceTempView("useritem")
        # row counts = 836006
        spark.sql('SELECT COUNT(*) FROM useritem')
        # item counts = 266414
        spark.sql('SELECT COUNT(DISTINCT itemId) FROM useritem')
        # user counts = 478235
        spark.sql('SELECT COUNT(DISTINCT userId) FROM useritem')
        # number of users who only rate for 1 item: 358615
        spark.sql(
            'SELECT COUNT(*) FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count == 1)')
        # select user who rate for > 1 items and items rated by > 1 user
        users = 'SELECT userId FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count > 1)'
        items = 'SELECT itemId FROM (SELECT DISTINCT COUNT(*) AS count, itemId FROM useritem GROUP BY itemId HAVING count > 1)'

        # after filter out the sparse part: 353553
        df = spark.sql('SELECT * FROM useritem WHERE userId IN (' + users + ')')  # AND itemId IN (' + items + ')')
        # df.write.csv('filtered_data')
    else:
        print('loading processed data')
        df = spark.read.csv("filtered_data").toDF('userId', 'itemId', 'rating', 'timestamp')
        df.show()

    print('count of filtered db: {0}'.format(df.count()))
    userIdIntMap = df.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
    itemIdIntMap = df.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()

    ratings = df.rdd.map(lambda d:Row(userId=userIdIntMap.get(d.userId), itemId=itemIdIntMap.get(d.itemId), rating=float(d.rating))).toDF()
    ratings.show()
    (training, test) = ratings.randomSplit([0.8, 0.2])
    training.write.csv('training')
    test.write.csv('test')
