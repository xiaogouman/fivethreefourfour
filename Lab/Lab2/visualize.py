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
rawdf = spark.read.csv("ratings_Sports_and_Outdoors.csv").toDF('userId', 'itemId', 'rating', 'timestamp')

# map userId and itemId to iteger
userIdIntMap = rawdf.rdd.map(lambda r: r.userId).distinct().zipWithUniqueId().collectAsMap()
itemIdIntMap = rawdf.rdd.map(lambda r: r.itemId).distinct().zipWithUniqueId().collectAsMap()
rawdf = rawdf.rdd.map(
    lambda d: Row(userId=userIdIntMap.get(d.userId), itemId=itemIdIntMap.get(d.itemId), rating=float(d.rating))).toDF()

# get ratings' count grouped by userId
counts = rawdf.groupBy(rawdf.userId).count()
counts_group = counts.rdd.map(lambda row:(row['count'], 1)).reduceByKey(lambda a, b: a+b).sortByKey(ascending=True).collect()
print(counts_group)
# musical instruments
#(1, 270914), (2, 39277), (3, 13015), (4, 5968), (5, 3226), (6, 1973), (7, 1230), (8, 786), (9, 569), (10, 430), (11, 356), (12, 247), (13, 192), (14, 143), (15, 121), (16, 96), (17, 95), (18, 58), (19, 64), (20, 51), (21, 46), (22, 45), (23, 34), (24, 31), (25, 25), (26, 22), (27, 9), (28, 21), (29, 15), (30, 12), (31, 14), (32, 12), (33, 6), (34, 3), (35, 8), (36, 8), (37, 5), (38, 4), (39, 5), (40, 8), (41, 1), (42, 4), (43, 2), (44, 1), (46, 3), (47, 4), (48, 5), (49, 3), (50, 3), (51, 4), (52, 3), (53, 1), (55, 5), (56, 1), (58, 1), (59, 2), (60, 1), (61, 2), (62, 3), (63, 1), (64, 3), (67, 3), (69, 1), (71, 2), (72, 2), (76, 1), (77, 1), (82, 2), (84, 1), (86, 2), (89, 2), (94, 1), (97, 1), (99, 1), (101, 1), (106, 1), (108, 1), (110, 2), (113, 1), (114, 1), (118, 1), (126, 1), (135, 1), (154, 1), (454, 1), (463, 1), (483, 1)]

# sports
#[(1, 1463787), (2, 288644), (3, 104732), (4, 48990), (5, 26588), (6, 16106), (7, 10328), (8, 7123), (9, 5053), (10, 3678), (11, 2834), (12, 2167), (13, 1711), (14, 1305), (15, 1081), (16, 851), (17, 727), (18, 592), (19, 484), (20, 407), (21, 363), (22, 324), (23, 268), (24, 227), (25, 184), (26, 193), (27, 182), (28, 147), (29, 115), (30, 111), (31, 118), (32, 72), (33, 75), (34, 82), (35, 65), (36, 68), (37, 63), (38, 54), (39, 41), (40, 33), (41, 32), (42, 27), (43, 28), (44, 30), (45, 25), (46, 29), (47, 19), (48, 22), (49, 19), (50, 17), (51, 12), (52, 11), (53, 18), (54, 10), (55, 11), (56, 10), (57, 16), (58, 16), (59, 11), (60, 7), (61, 7), (62, 6), (63, 9), (64, 4), (65, 7), (66, 4), (67, 4), (68, 10), (69, 8), (70, 3), (71, 5), (72, 3), (73, 5), (74, 4), (75, 6), (76, 2), (77, 6), (78, 3), (79, 4), (80, 6), (81, 2), (82, 3), (83, 3), (84, 3), (85, 1), (87, 1), (88, 3), (89, 1), (90, 2), (91, 2), (92, 1), (94, 6), (95, 1), (97, 1), (98, 1), (99, 3), (101, 2), (103, 1), (104, 2), (105, 1), (106, 1), (109, 4), (111, 1), (112, 1), (113, 1), (115, 2), (117, 1), (119, 3), (120, 2), (122, 1), (123, 1), (124, 1), (129, 1), (131, 1), (143, 1), (147, 1), (153, 1), (155, 1), (156, 1), (159, 1), (168, 1), (194, 1), (224, 1), (254, 1), (403, 1)]


# counts_keys = [count[0] for count in counts_group]
# counts_values = [count[1] for count in counts_group]

# plt.bar(counts_keys, counts_values)
# plt.show()

############## histogram #################
fig, axes = plt.subplots(2, 1)
ax0, ax1 = axes[0], axes[1]
# all users
counts_list = counts.rdd.map(lambda row:row['count']).collect()
all_counts_list = [x for x in counts_list if x < 200]
ax0.hist(all_counts_list, 100, density=True)

# remove users who only rate one item
counts_remove_one = [x for x in counts_list if x > 1 and x < 200]
ax1.hist(counts_remove_one, 100, density=True)
plt.show()
