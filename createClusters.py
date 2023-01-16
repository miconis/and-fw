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


def tsne_plot(cluster_name, cluster_data, n_components, perplexity, early_exaggeration, learning_rate, n_iter, init, method):
    data_subset = cluster_data.select(col("gtid"), col("topicDistribution"))

    x = np.array([ele["topicDistribution"] for ele in data_subset.collect()])
    y = [ele["gtid"] for ele in data_subset.collect()]

    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                init=init,
                method=method)  # TODO check parameters for TSNE

    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 19),
                    data=df).set(title="%s data T-SNE projection" % cluster_name)
    plt.show()
    return


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

authors = sc.textFile("../datasets/Aminer-WhoIsWho (na-v3)/processed/authors_extracted.json").map(json.loads).map(lambda a: (lnfi(a['name']), a))

# iterate over clustering keys to process one cluster at a time
for clustering_key in authors.map(lambda a: a[0]).filter(lambda a: a != '').distinct().collect():
    # isolate the cluster of authors (based on LNFI)
    cluster = authors\
        .filter(lambda a: clustering_key in a[0])\
        .map(lambda a: a[1])\
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
    lda = LDA(k=4,                              # number of topics:the larger is the dataset, the greater is the number of topics, only if the dataset is representative of a diverse collection. The optimal number of topics is estimated by calculating topic coherence but it's time consuming, it is better to determine it heuristically.
              maxIter=100,                      # maximum number of iteration to stop the process
              optimizer='online',               # inference algorithm used to estimate the LDA model
              learningOffset=5,                 # learning parameter that downweights early iterations. Larger values make early iterations count less. 1024 is the default
              topicConcentration=5,             # beta (topic-word density): with a high beta value, topics are assumed to be made up of most words in the fixed-sized vocabulary and this results in a more specific word mixture for each topic
              docConcentration=[50],            # alpha (document-topic density): with a high alpha value, documents are assumed to contain more topics. Initially the alpha parameter can be set to a real number value divided by the number of topics
              learningDecay=0.51,               # learning rate set as an exponential decay rate. This should be between (0.5, 1.0] to guarantee asymptotic convergence. Starts training with a large learning rate and then slowly decaying it until local minima is obtained
              optimizeDocConcentration=True)    # indicates if the docConcentration (alpha) parameter can be optimized during the training

    model = lda.fit(rescaledCluster)

    transformedCluster = model.transform(rescaledCluster)

    # remove useless columns
    transformedCluster = transformedCluster.drop(*("abstract", "words", "filtered", "cv_features", "features"))

    tsne_plot(cluster_name=clustering_key,          # name of the cluster
              cluster_data=transformedCluster,      # data to be processed
              n_iter=1000,                          # number of iterations
              early_exaggeration=25,                # controls how tight natural clusters are in the embedding space and how much space will be between them. Larger value, larger space (default is 12.0)
              init='pca',                           # could be in the set {'random', 'pca'}. PCA initialization is more globbally stable than random initialization
              method='exact',                       # gradient calculation algorithm. Could be in the set {'barnes-hut', 'exact'}. The first has lower complexity, the second has higher complexity. Exact does not scale well with the number of samples but it's more precise.
              learning_rate="auto",                 # usually in range [10, 1000]. If too low, points compressed in a cloud with few outliers. If too high, data look like a ball with any point approximately equidistant from its nearest neighbours. 'auto' option sets the learning_rate to `max(N / early_exaggeration / 4, 50)` where N is the sample size
              perplexity=5,                         # larger datasets requires larger perplexity. The value should be between 5 and 50 and it should be lower than the number of samples
              n_components=2                        # size of the embedding space
              )

    exit()
