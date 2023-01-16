def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import json
import os
import pickle
import re
from multiprocessing import freeze_support
import csv
import nltk
import numpy as np
import pandas as pd
import pyLDAvis
import seaborn as sns
import spacy
import tqdm
from gensim import models, corpora
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from pyLDAvis import gensim
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from mpl_toolkits import mplot3d


def list_to_array(x, k):
    res = [0] * k
    for i in x:
        res[i[0]] = i[1]
    return res


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True is to remove punctuations


def lemmatization(texts, allowed_postags=['VERB', 'NOUN', 'ADJ', 'ADV', 'PROPN', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PRON', 'PUNCT', 'SCONJ', 'X', 'SYM']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def create_corpus(papers, dictionary, text_field):
    # remove punctuation and convert to lowercase
    papers["corpus"] = \
        papers[text_field] \
        .map(lambda x: re.sub('[,\.!?%$()0123456789:;]', '', x))  # punctuations
    papers["corpus"] = \
        papers["corpus"] \
        .map(lambda x: x.lower())  # lowercase

    data = papers["corpus"].values.tolist()  # collect paper texts
    data_words = list(sent_to_words(data))  # collect paper words

    bigram = models.Phrases(data_words, min_count=5, threshold=100)  # higher parameters' values, harder for words to be combined
    bigram_mod = models.phrases.Phraser(bigram)
    data_words_nostops = remove_stopwords(data_words)  # remove stopwords
    # data cleaning
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]  # form bigrams
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'PROPN', 'X'])

    texts = data_lemmatized  # create corpus

    corpus = [dictionary.doc2bow(text) for text in texts]  # compute term document frequency
    tfidf = models.TfidfModel(corpus)
    return tfidf[corpus]


def inject_topics(join_res):
    res = join_res[1][0]
    res['topics'] = join_res[1][1]
    return res


def filter_author_name(name, x):
    for a in x['aids']:
        if name in a['key']:
            return True
    return False


def tsne_plot(element, n_components=2, perplexity=25, early_exaggeration=12, learning_rate=0.51, n_iter=1000, init='pca', method='barnes_hut', random_state=0):
    plt.figure()
    docs = list(map(lambda e: e['topics'], element))

    x = np.array(docs)
    y = list(map(lambda e: e['aid'], element))  # choose the label for each doc

    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                init=init,
                method=method,
                random_state=random_state)

    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(set(y))),
                    data=df).set(title="LDA docs T-SNE projection")
    # plt.savefig("%s/tsne_plot_k%s_a%s_b%s.png" % (output_base_path, k, alpha, beta))
    plt.show()
    return


input_path = "../datasets/Aminer-WhoIsWho (na-v3)/processed/pubs_enriched.json"
topics_path = "../results/lda_documents"
dictionary_path = "../results/dictionary_no_below5_no_above0.7"
lda_model_path = "../results/lda_model_k9_aauto_bauto"
text_field = "abstract"
generate_topics = True

conf = SparkConf().setAppName('AMiner-WhoIsWho Stats').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sqlContext = SQLContext(sc)

nltk.download('stopwords')
stop_words = stopwords.words('english')
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])  # spacy pipeline: tokenizer, lemmatizer, tagger

papers = sc.textFile(input_path) \
    .map(json.loads) \
    .filter(lambda x: x[text_field]) \
    .filter(lambda x: filter_author_name("hao_du", x))

lda_docs = []
if generate_topics:
    lda_model = LdaModel.load(lda_model_path)
    dictionary = corpora.Dictionary.load(dictionary_path)

    papers_pd = papers.toDF().toPandas()

    corpus = create_corpus(papers=papers_pd, dictionary=dictionary, text_field=text_field)
    lda_docs = lda_model[corpus]  # compute the distribution of topics in each document
    lda_docs = [dict(id=papers_pd['id'].iloc[i], topics=list_to_array(lda_docs[i][0], 9)) for i in range(0, len(lda_docs))]
    with open(topics_path, "wb") as fp:
        pickle.dump(lda_docs, fp)
else:
    with open(topics_path, "rb") as fp:
        lda_docs = pickle.load(fp)

topics = sc.parallelize(lda_docs).map(lambda x: (x['id'], x['topics']))

papers = papers.map(lambda x: (x['id'], x)).join(topics).map(inject_topics)

authors = papers.flatMap(lambda p: [dict(aid=a['aid'], topics=p['topics']) for a in p['aids']])

tsne_plot(authors.collect())
