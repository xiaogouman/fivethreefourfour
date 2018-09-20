import re
import math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import numpy as np
import logging

'''
Name: Zuo Shumman
StudentId: E0336049

Remarks: I use 'reviews_Kindle_Store_5.json' as the dataset, which is about 800M big, due to limited
resouces of my PC.

Top 20 scores(cosine similarity):
====================
DocId,Score
ARW2ZC35RT9DP-B00EUGVXJ2,0.5373703023251719
A1QQWCV32TRER7-B005VFXLIW,0.4956585034969728
A1B7DR0K0VCMM-B008IGHGXW,0.4943300928461792
A3IC05KV0TWHEE-B00FKKWQXY,0.49368696018745556
A323H7CL7G313K-B00G2GMRCU,0.48673346112366367
A2I6Z9AKM51CZI-B00H9KEAKA,0.4682728326842841
A36AIW5KPEKEHQ-B0070TFYSC,0.4663152059875413
A3MCGUE3S4HPL5-B00ASPKRSC,0.45924916044852593
AZMMFZKJB8PV6-B005NWIAAW,0.4536083110108068
A1I53V3YF42546-B00C2YRAQE,0.4521820454419772
AFBLYN4LJ4UQH-B008QGJ1K0,0.4521369839687979
A3O5UR6NHR4MRP-B00CX6Z88I,0.45082241314166005
A3O5UR6NHR4MRP-B00CKVE6U2,0.45058593855811313
A23ILLE19GTILU-B00B77TD6M,0.45022804701790187
A2ZTY51F0BEQ9P-B00IB4YWX8,0.44997162370324245
A1DZ5U6YLKI2HZ-B00B60R6W8,0.449012777989839
ARGVN3VQ75W77-B00IRLFJ96,0.4385550146049371
A1FSE5S5EKD58F-B00EEE52DI,0.4349192123001444
AW6BVPL0TZPQU-B00BGY7XX6,0.43446520160712876
A20JCYF5CXS6PX-B00CJ2BR6S,0.43441960591123024
====================
'''

# step 1: return word count in format of ((doc_id, word), count)
def get_doc_word_counts(docs, stopwords):
    doc_word_tuples = docs \
        .flatMap(lambda doc: [(doc[0], word.lower()) for word in re.sub(r"[^a-zA-Z0-9]+", ' ', doc[1]).split(' ')])
    doc_word_counts = doc_word_tuples \
        .filter(lambda doc_word: doc_word[1] != '' and doc_word[1] not in stopwords) \
        .map(lambda doc_word: ((doc_word[0], doc_word[1]), 1)) \
        .reduceByKey(lambda n1, n2: n1 + n2)
    return doc_word_counts


# step 2: calculate tf-tdf for every word for every doc, return in format of ((doc_id, word), count)
def tf_idf(tf, df, n):
    return (1 + math.log(tf, 10)) * math.log(n / df, 10)


def calculate_tf_idf(word_group, n):
    count = len(word_group[1])
    word = word_group[0]
    return [((doc_count[0], word), tf_idf(int(doc_count[1]), count, n))
            for doc_count in word_group[1]]


def get_tf_idf(doc_word_counts, n):
    return doc_word_counts \
        .map(lambda doc_word_count: (doc_word_count[0][1], (doc_word_count[0][0], doc_word_count[1]))) \
        .groupByKey().flatMap(lambda word_group: calculate_tf_idf(word_group, n))


# step 3: normalized tg-idf for every word in every doc,
def calculate_normalized_tf_idf(doc_group):
    doc_id = doc_group[0]
    word_tf_idf = doc_group[1]
    sqrt_sum = math.sqrt(sum([each_tf_idf[1] * each_tf_idf[1] for each_tf_idf in word_tf_idf]))
    return [((doc_id, each_tf_idf[0]), each_tf_idf[1] / sqrt_sum) for each_tf_idf in word_tf_idf]


def get_normalized_tf_idf(tf_idf):
    return tf_idf \
        .map(lambda doc_word_tf_idf: (doc_word_tf_idf[0][0], (doc_word_tf_idf[0][1], doc_word_tf_idf[1]))) \
        .groupByKey().flatMap(lambda doc_group: calculate_normalized_tf_idf(doc_group))


# step 4: calculate relevance score
def get_relevance_score(normalized_tf_idf, query):
    normalized_query = 1/math.sqrt(len(query))
    return normalized_tf_idf \
        .map(lambda each_tf_idf: (each_tf_idf[0][0], each_tf_idf[1]*normalized_query if each_tf_idf[0][1] in query else 0)) \
        .reduceByKey(lambda a,b: a+b)


# step 5 return scores sorted by score
def get_sorted_scores(scores, k):
    return scores.sortBy(lambda score: score[1], False).take(k)


def main():
    conf = SparkConf()\
        .setAppName('book-reviews')
    sc = SparkContext(conf=conf)
    spark = SparkSession \
        .builder \
        .getOrCreate()
    my_logger = logging.getLogger('BookReviews')
    my_dir = '/Users/xiaogouman/Documents/masters/CS5344/Lab/Lab1/'

    query = set(sc.textFile(my_dir+'query.txt').flatMap(lambda l: re.split(r'[^\w]+', l))
                .map(lambda s: s.lower()).collect())

    stopwords = set(sc.textFile(my_dir+'stopwords.txt')
                    .flatMap(lambda l: re.split(r'[^\w]+', l))
                    .map(lambda s: s.lower()).collect())

    # read json using sparksql and map to document type
    reviews = spark.read.json(my_dir+'reviews_Kindle_Store_5.json')

    docs = reviews.rdd.map(lambda doc: (doc.reviewerID+'-'+doc.asin, doc.reviewText + ' ' + doc.summary))
    n = docs.count()
    k = 20

    # step 1
    my_logger.info('Step 1 starts')
    doc_word_counts = get_doc_word_counts(docs, stopwords)
    my_logger.info('Step 1 ends')
    #doc_word_counts.saveAsTextFile('docWordCounts')

    # step 2
    my_logger.info('Step 2 starts')
    tf_idf = get_tf_idf(doc_word_counts, n)
    my_logger.info('Step 2 ends')
    #tf_idf.saveAsTextFile('tfIdf')

    # step 3
    my_logger.info('Step 3 starts')
    normalized_tf_idf = get_normalized_tf_idf(tf_idf)
    my_logger.info('Step 3 ends')
    #normalized_tf_idf.saveAsTextFile('normalizedTfIdf')

    # step 4
    my_logger.info('Step 4 starts')
    scores = get_relevance_score(normalized_tf_idf, query)
    my_logger.info('Step 4 ends')
    #scores.saveAsTextFile('scores')

    # step 5
    my_logger.info('Step 5 starts')
    sorted_scores = get_sorted_scores(scores, k)
    my_logger.info('Step 5 ends')

    for score in sorted_scores:
        print('DocId: {0}, Score: {1}'.format(score[0], score[1]))

    with open("scores.txt", 'w') as f:
        np.savetxt(f, sorted_scores, delimiter=',', fmt='%s')

    sc.stop()


if __name__ == '__main__':
    main()
