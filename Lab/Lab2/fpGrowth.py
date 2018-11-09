from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql import Row
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.window import Window as W

spark = SparkSession \
    .builder \
    .appName("Lab2") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# The dataset we have used for training is musical instrument metadata
df = spark.read.json("meta_Musical_Instruments.json")

df.createOrReplaceTempView('musical')
spark.sql('select COUNT(*) as total_count from musical').show()
spark.sql('select COUNT(*) as null_count from musical where related is NULL or related.bought_together is NULL').show()

# We have observed for some metadata ths asin, related and bought_together fields are Null. We have filtered out those fields.
df = df.filter(df.asin.isNotNull()).filter(df.related.isNotNull()).filter(df.related.also_viewed.isNotNull())

# We have set the mimimum number of bought_together items to be 2. So in total, there will be at least 2 items in a row.
rawData= df.rdd.filter(lambda row: len(row.related.also_viewed) >= 1)

# We have reconstruced the train data frame work. Each reconstructed row will contain asin item and item which a user bought together.
trainDF = rawData.map(lambda row: Row(bought=list(set([row.asin] + row.related.also_viewed)))).toDF()

# We have used FPGrowth algorithm.
fpGrowth = FPGrowth(itemsCol="bought", minSupport=0.001, minConfidence=0.5)
#fpGrowth = FPGrowth(itemsCol="bought", minSupport=0.0000005, minConfidence=0.0000006)
model = fpGrowth.fit(trainDF)

# We have generated the association rules between items. The item count for antecedent and consequent has been restricted to 1. 
# The dataframe shows the confidence rate from antecedent to consequent.
association = model.associationRules.rdd.filter(lambda row: len(row.antecedent) == 1 and len(row.consequent) == 1)
association_item = association.toDF()
association_item = association_item.orderBy(association_item.confidence, ascending=False)

# We have used a map to store the antecedent -> consequent relationship. The consequent are stored in list with the highest confidence rate one in the front.
associationMap = {}
for association_data in association_item.rdd.collect():
    
    antecedent = association_data.antecedent[0]
    consequent = association_data.consequent
    
    if antecedent not in associationMap:
        associationMap[antecedent] = consequent
    else:
        associationMap[antecedent] = associationMap[antecedent] + consequent






# For testing and conversion rate calculation, we will use the same testing dataset used by collaborative filtering, which is seperated by ratings dataset.
# We have used a ratings map to store the user and the items rated by the user. 
rawRatings = spark.read.csv("ratings_Musical_Instruments.csv").toDF('userId', 'itemId', 'rating', 'timestamp')

Ratings = {}
for user_data in rawRatings.rdd.collect():

    userId = user_data.userId
    item = user_data.itemId

    if userId not in Ratings:
        Ratings[userId] = [item]
    else:
        Ratings[userId] = Ratings[userId] + [item]


# We have used a test list to store the (item,user).
testDF = spark.read.csv("test.csv").toDF('userId', 'itemId', 'rating')
testRDD = testDF.rdd.map(lambda row: Row(userId = [row.userId],itemId = row.itemId))
test_list = [list(row) for row in testRDD.collect()]


# The count of the test size
count = len(test_list)

# The count of the match of prediction with reality
total = 0

# top K items
k=5


# Iterate through test set from first data to last data
for i in range(0,count):
    
    # Get the asin item of data, which will be used to as antecedent in the association map. 
    asin= test_list[i][0]
    # Get the user id of data, which will be used to retrived rated items in the association map.
    userId = test_list[i][1][0]
  

    if asin in associationMap:
        
        consequent_list = associationMap[asin]
        l = len(consequent_list)

        # Check how many items could be recommended to the user
        if k<= l:
            consequent_items = k
        else:
            consequent_items = l

        for j in range(0,consequent_items):
            # check if the recommended item in user's rated item set. If yes, add 1 to count and break the loop.
            if consequent_list[j] in Ratings[userId]:
                total = total + 1
                break

print (total*1.0/len(test_list))




















