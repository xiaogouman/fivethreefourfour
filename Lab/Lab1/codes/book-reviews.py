import re
import math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import numpy as np
import logging


# step 1: return word count in format of ((doc_id, word), count)
def get_doc_word_counts(docs, stopwords):
    doc_word_tuples = docs \
        .flatMap(lambda doc: [(doc[0], word.lower()) for word in re.sub(r"[^a-zA-Z0-9]+", ' ', doc[1]).split(' ')])
    doc_word_counts = doc_word_tuples \
        .filter(lambda doc_word: doc_word[1] != '' and doc_word[1] not in stopwords) \
        .mapPartitions(lambda doc_words: [((doc_word[0], doc_word[1]), 1) for doc_word in doc_words]) \
        .reduceByKey(lambda n1, n2: n1 + n2)
    return doc_word_counts


# step 2: calculate tf-tdf for every word for every doc, return in format of ((doc_id, word), count)
def tf_idf(tf, df, n):
    return (1 + math.log(tf,10)) * math.log(n / df, 10)


def calculate_tf_idf(word_group, n):
    count = len(word_group[1])
    word = word_group[0]
    return [((doc_count[0], word), tf_idf(int(doc_count[1]), count, n))
            for doc_count in word_group[1]]


def get_tf_idf(doc_word_counts, n):
    return doc_word_counts \
        .mapPartitions(
        lambda doc_word_count_partition:
        [(doc_word_count[0][1], (doc_word_count[0][0], doc_word_count[1]))
         for doc_word_count in doc_word_count_partition]) \
        .groupByKey().flatMap(lambda word_group: calculate_tf_idf(word_group, n))


# step 3: normalized tg-idf for every word in every doc,
def calculate_normalized_tf_idf(doc_group):
    doc_id = doc_group[0]
    word_tf_idf = doc_group[1]
    sqrt_sum = math.sqrt(sum([each_tf_idf[1] * each_tf_idf[1] for each_tf_idf in word_tf_idf]))
    return [((doc_id, each_tf_idf[0]), each_tf_idf[1] / sqrt_sum) for each_tf_idf in word_tf_idf]


def get_normalized_tf_idf(tf_idf):
    return tf_idf \
        .mapPartitions(
        lambda doc_word_tf_idf_partition:
        [(doc_word_tf_idf[0][0], (doc_word_tf_idf[0][1], doc_word_tf_idf[1]))
         for doc_word_tf_idf in doc_word_tf_idf_partition]) \
        .groupByKey().flatMap(lambda doc_group: calculate_normalized_tf_idf(doc_group))


# step 4: calculate relevance score
# doc_tf_idf is of format (word, normalized_tf_idf)
def calculate_score(doc_tf_idf, query):
    v1 = np.array([each_tf_idf[1] for each_tf_idf in doc_tf_idf])
    v2 = np.array([1 if each_tf_idf[0] in query else 0 for each_tf_idf in doc_tf_idf])
    top_part = v1.dot(v2.transpose())
    if top_part == 0:
        return 0
    else:
        return top_part / (np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2)))


def get_relevance_score(normalized_tf_idf, query):
    return normalized_tf_idf \
        .mapPartitions(
        lambda each_tf_idf_partition:
        [(each_tf_idf[0][0], (each_tf_idf[0][1], each_tf_idf[1]))
         for each_tf_idf in each_tf_idf_partition]) \
        .groupByKey() \
        .mapPartitions(
        lambda doc_group_partition:
        [(doc_group[0], calculate_score(doc_group[1], query))
         for doc_group in doc_group_partition])


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
    #reviews = spark.read.json(my_dir+'bookreviews-short.json')

    docs = reviews.rdd.map(lambda doc: (doc.reviewerID+doc.asin, doc.reviewText + ' ' + doc.summary))
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
