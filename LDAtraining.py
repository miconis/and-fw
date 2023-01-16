import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.manifold import TSNE


def lnfi(s):
    fullname = s.split(" ")
    if len(fullname) == 1:
        return str(''.join(ch for ch in "".join(fullname).lower() if ch.isalnum()))
    if len(fullname) == 2:
        try:
            return str(''.join(ch for ch in fullname[0][0].lower() if ch.isalnum())) + str(''.join(ch for ch in fullname[1].lower() if ch.isalnum()))
        except:
            return str(''.join(ch for ch in fullname[1].lower() if ch.isalnum()))
    if len(fullname) > 2:
        surname = [fullname[len(fullname) - 1]]
        name = fullname[0:len(fullname) - 1]
        res = " ".join(name).lower()[0] + " ".join(surname).lower()
        return str(''.join(ch for ch in res if ch.isalnum()))


conf = SparkConf().setAppName('AMiner-WhoIsWho Stats').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sqlContext = SQLContext(sc)
StopWords = stopwords.words("english")
regexTokenizer = RegexTokenizer(inputCol="abstract", outputCol="words", pattern="\\W")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
countVectorizer = CountVectorizer(inputCol="words", outputCol="cv_features", vocabSize=3, minDF=2.0)
idf = IDF(inputCol="cv_features", outputCol="features")

authors = sc.textFile("../datasets/Aminer-WhoIsWho (na-v3)/processed/authors_extracted.json").map(json.loads)

cluster = authors \
    .filter(lambda a: 'jwang' in lnfi(a['name'])) \
    .filter(lambda a: a['publication']['abstract'] != "")\
    .toDF()

# tokenize the cluster (split abstracts into words and remove stopwords)
tokenizedCluster = stopwordsRemover\
    .transform(
        regexTokenizer
        .transform(
               cluster.withColumn('abstract', col('publication.abstract'))
        )
    )

# TODO ADD PHASE: REMOVE RARE AND MOST COMMON TOKENS

# vectorize the cluster (convert tokens into topic distribution vector)
cvModel = countVectorizer.fit(tokenizedCluster)
countVectorizedCluster = cvModel.transform(tokenizedCluster)

idfModel = idf.fit(countVectorizedCluster)
rescaledCluster = idfModel.transform(countVectorizedCluster)

#fine tuning alpha and beta:
    #chose an alpha<1 if it is expected that the distribution of topics in each document is sparse (each document contains only a few topics)
    #chose a beta<1 if it is expected that topics favour some words
    #if you don't know how to choose:
        #choose alpha between [0.05, 0.1, 0.5, 1, 5, 10, 50]
        #choose beta between [0.01, 0.05, 0.1, 1, 5, 10]
        #train LDA with alpha and beta (other parameters have to be fixed)
        #calculate perplexity score
        #choose alpha and beta pair with the minimum perplexity score
lda = LDA(k=5,                             # number of topics:the larger is the dataset, the greater is the number of topics, only if the dataset is representative of a diverse collection. The optimal number of topics is estimated by calculating topic coherence but it's time consuming, it is better to determine it heuristically.
          maxIter=20,                       # maximum number of iteration to stop the process
          docConcentration=[50],            # alpha (document-topic density): with a high alpha value, documents are assumed to contain more topics. Initially the alpha parameter can be set to a real number value divided by the number of topics
          topicConcentration=10,            # beta (topic-word density): with a high beta value, topics are assumed to be made up of most words in the fixed-sized vocabulary and this results in a more specific word mixture for each topic
          optimizer='online',               # inference algorithm used to estimate the LDA model
          learningOffset=5,                 # learning parameter that downweights early iterations. Larger values make early iterations count less. 1024 is the default
          learningDecay=0.51,               # learning rate set as an exponential decay rate. This should be between (0.5, 1.0] to guarantee asymptotic convergence. Starts training with a large learning rate and then slowly decaying it until local minima is obtained
          optimizeDocConcentration=True)    # indicates if the docConcentration (alpha) parameter can be optimized during the training

model = lda.fit(rescaledCluster)

print(model.logPerplexity(rescaledCluster).as_integer_ratio())

model.save("../lda_model_k5_iter20_a50_b10_online_lo5_ld051")
