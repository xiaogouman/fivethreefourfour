from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark.sql.functions import countDistinct
import pandas as pd
import os
import matplotlib.pyplot as plt

spark = SparkSession.builder.master("local").appName("Collborative Filering").getOrCreate()

# load csv
rawdf = spark.read.csv("ratings_Musical_Instruments.csv").toDF('userId', 'itemId', 'rating', 'timestamp')

# map userId and itemId to iteger
userIdIntMap = rawdf.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
itemIdIntMap = rawdf.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()
rawdf = rawdf.rdd.map(
    lambda d: Row(userId=userIdIntMap.get(d.userId), itemId=itemIdIntMap.get(d.itemId), rating=float(d.rating))).toDF()

# get ratings' count grouped by userId
counts = rawdf.groupBy(rawdf.userId).count()
#counts_group = counts.rdd.map(lambda row:(row['count'], 1)).reduceByKey(lambda a, b: a+b).sortByKey(ascending=True).collect()
# print(counts_group)
#[(1, 358615), (2, 62806), (3, 22657), (4, 11279), (5, 6482), (6, 3996), (7, 2669), (8, 1907), (9, 1341), (10, 1030), (11, 769), (12, 611), (13, 503), (14, 446), (15, 358), (16, 285), (17, 250), (18, 212), (19, 184), (20, 148), (21, 151), (22, 140), (23, 101), (24, 106), (25, 88), (26, 65), (27, 67), (28, 61), (29, 60), (30, 47), (31, 41), (32, 47), (33, 38), (34, 23), (35, 42), (36, 35), (37, 27), (38, 16), (39, 17), (40, 19), (41, 18), (42, 19), (43, 19), (44, 13), (45, 19), (46, 25), (47, 13), (48, 7), (49, 19), (50, 15), (51, 14), (52, 11), (53, 9), (54, 11), (55, 9), (56, 7), (57, 9), (58, 7), (59, 7), (60, 15), (61, 12), (62, 7), (63, 6), (64, 10), (65, 6), (66, 6), (67, 5), (68, 4), (69, 7), (70, 5), (71, 3), (72, 1), (73, 6), (74, 2), (75, 3), (76, 6), (77, 1), (78, 4), (79, 4), (80, 2), (81, 3), (82, 2), (83, 1), (84, 5), (85, 3), (86, 2), (87, 1), (88, 1), (89, 2), (92, 2), (94, 2), (96, 1), (97, 2), (98, 1), (99, 2), (100, 4), (101, 2), (102, 1), (103, 6), (104, 2), (105, 2), (106, 5), (107, 1), (108, 2), (109, 2), (110, 1), (111, 1), (113, 3), (114, 1), (117, 1), (120, 2), (121, 1), (122, 2), (123, 1), (124, 2), (127, 1), (128, 2), (129, 1), (131, 1), (133, 1), (134, 2), (135, 1), (137, 1), (142, 3), (143, 1), (144, 1), (146, 1), (148, 1), (154, 1), (164, 1), (168, 1), (170, 1), (172, 2), (174, 1), (175, 1), (176, 1), (178, 1), (179, 1), (180, 2), (182, 1), (183, 1), (184, 1), (186, 1), (195, 1), (199, 2), (200, 2), (210, 1), (218, 1), (243, 1), (257, 1), (265, 1), (269, 1), (280, 1), (305, 1), (317, 1), (318, 1), (339, 1), (343, 1), (347, 1), (409, 1), (427, 1), (471, 1), (489, 1), (713, 1), (1126, 1)]

# counts_keys = [count[0] for count in counts_group]
# counts_values = [count[1] for count in counts_group]
#
# plt.bar(counts_keys, counts_values)
# plt.show()
fig, axes = plt.subplots(2, 1)
ax0, ax1 = axes[0], axes[1]
# all users
counts_list = counts.rdd.map(lambda row:row['count']).collect()
all_counts_list = [x for x in counts_list if x < 1000]
ax0.hist(all_counts_list, 100, density=True)

# remove users who only rate one item
counts_remove_one = [x for x in counts_list if x > 2 and x < 1000]
ax1.hist(counts_remove_one, 100, density=True)
plt.show()






# # row counts = 836006
# spark.sql('SELECT COUNT(*) FROM useritem')
# # item counts = 266414
# spark.sql('SELECT COUNT(DISTINCT itemId) FROM useritem')
# # user counts = 478235
# spark.sql('SELECT COUNT(DISTINCT userId) FROM useritem')
# # number of users who only rate for 1 item: 358615
# spark.sql(
#     'SELECT COUNT(*) FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count == 1)')
# # select user who rate for > 1 items and items rated by > 1 user
# users = 'SELECT userId FROM (SELECT DISTINCT COUNT(*) AS count, userId FROM useritem GROUP BY userId HAVING count > 1)'
# items = 'SELECT itemId FROM (SELECT DISTINCT COUNT(*) AS count, itemId FROM useritem GROUP BY itemId HAVING count > 1)'
#
# # after filter out the sparse part: 353553
# df = spark.sql('SELECT * FROM useritem WHERE userId IN (' + users + ')')  # AND itemId IN (' + items + ')')
#
#
#
# print('count of filtered db: {0}'.format(df.count()))
# userIdIntMap = df.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
# itemIdIntMap = df.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()

# ratings = df.rdd.map(lambda d:Row(userId=userIdIntMap.get(d.userId), itemId=itemIdIntMap.get(d.itemId), rating=float(d.rating))).toDF()
# (training, test) = ratings.randomSplit([0.8, 0.2])


# df = spark.read.csv("filtered_data").toDF('userId', 'itemId', 'rating', 'timestamp')
# df.show()