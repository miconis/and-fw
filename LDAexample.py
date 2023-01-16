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


conf = SparkConf().setAppName('AMiner-WhoIsWho Stats').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sqlContext = SQLContext(sc)
StopWords = stopwords.words("english")
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
countVectorizer = CountVectorizer(inputCol="words", outputCol="cv_features", vocabSize=20, minDF=1.0, maxDF=5, minTF=1.0)
idf = IDF(inputCol="cv_features", outputCol="features")


def tsne_plot(cluster_name, cluster_data, n_components, perplexity, early_exaggeration, learning_rate, n_iter, init, method):
    data_subset = cluster_data.select(col("label"), col("topicDistribution"))

    x = np.array([ele["topicDistribution"] for ele in data_subset.collect()])
    y = [ele["label"] for ele in data_subset.collect()]

    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                init=init,
                method=method)

    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title="%s data T-SNE projection" % cluster_name)
    plt.show()
    return


documents = sc.parallelize([
    dict(id=1, label="food", text="I ate a banana and spinach smoothie for breakfast."),
    dict(id=2, label="food", text="I like to eat broccoli and bananas."),
    dict(id=3, label="animals", text="Chinchillas and kittens are cute."),
    dict(id=4, label="animals", text="My sister adopted a kitten yesterday."),
    dict(id=5, label="mix", text="Look at this cute hamster munching on a piece of broccoli")])\
    .toDF()

# tokenize the cluster (split abstracts into words and remove stopwords)
tokenizedCluster = stopwordsRemover\
    .transform(
        regexTokenizer
        .transform(documents)
    )

# vectorize the cluster (convert tokens into topic distribution vector)
cvModel = countVectorizer.fit(tokenizedCluster)
countVectorizedCluster = cvModel.transform(tokenizedCluster)

idfModel = idf.fit(countVectorizedCluster)
rescaledCluster = idfModel.transform(countVectorizedCluster)

lda = LDA(k=2,                              # number of topics:the larger is the dataset, the greater is the number of topics, only if the dataset is representative of a diverse collection. The optimal number of topics is estimated by calculating topic coherence but it's time consuming, it is better to determine it heuristically.
          maxIter=20,                       # maximum number of iteration to stop the process
          docConcentration=[0.1],           # alpha (document-topic density): with a high alpha value, documents are assumed to contain more topics. Initially the alpha parameter can be set to a real number value divided by the number of topics
          topicConcentration=1,             # beta (topic-word density): with a high beta value, topics are assumed to be made up of most words in the fixed-sized vocabulary and this results in a more specific word mixture for each topic
          optimizer='online',               # inference algorithm used to estimate the LDA model
          learningOffset=1024.0,            # learning parameter that downweights early iterations. Larger values make early iterations count less. 1024 is the default
          learningDecay=0.51,               # learning rate set as an exponential decay rate. This should be between (0.5, 1.0] to guarantee asymptotic convergence. Starts training with a large learning rate and then slowly decaying it until local minima is obtained
          optimizeDocConcentration=True)    # indicates if the docConcentration (alpha) parameter can be optimized during the training


model = lda.fit(rescaledCluster)

transformed = model.transform(rescaledCluster)

transformed.select(col("id"), col("text"), col("features"), col("topicDistribution")).show(truncate=False)

for topic in model.describeTopics().collect():
    print(topic)

print(model.vocabSize())

# tsne_plot(cluster_name="sentences",             # name of the cluster
#           cluster_data=transformed,             # data to be processed
#           n_iter=250,                           # number of iterations
#           early_exaggeration=25,                # controls how tight natural clusters are in the embedding space and how much space will be between them. Larger value, larger space (default is 12.0)
#           init='pca',                           # could be in the set {'random', 'pca'}. PCA initialization is more globbally stable than random initialization
#           method='exact',                       # gradient calculation algorithm. Could be in the set {'barnes-hut', 'exact'}. The first has lower complexity, the second has higher complexity. Exact does not scale well with the number of samples but it's more precise.
#           learning_rate="auto",                 # usually in range [10, 1000]. If too low, points compressed in a cloud with few outliers. If too high, data look like a ball with any point approximately equidistant from its nearest neighbours. 'auto' option sets the learning_rate to `max(N / early_exaggeration / 4, 50)` where N is the sample size
#           perplexity=4,                         # larger datasets requires larger perplexity. The value should be between 5 and 50 and it should be lower than the number of samples
#           n_components=2                        # size of the embedding space
# )

data_subset = transformed.select(col("label"), col("topicDistribution"))

x = np.array([ele["topicDistribution"] for ele in data_subset.collect()])
y = [ele["label"] for ele in data_subset.collect()]

df = pd.DataFrame()
df["y"] = y
df["comp-1"] = [item[0] for item in x]
df["comp-2"] = [item[1] for item in x]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 3),
                data=df).set(title="LDA data projection")
plt.show()


#perplexity is a statistical measure of how well a probability model predicts a sample. The model with the lowest perplexity is considered to be the best.
