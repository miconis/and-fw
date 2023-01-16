#LDA: Latent Dirichlet Allocation
#PARAMETERS OF LDA:
#alpha: document-topic density, high alpha=documents made up of more topics, more specific topic distribution per document
#beta: topic-word density, high beta=topics made up most of the words, more specific word distribution per topic
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


def plot_wordcloud(x, output_path='../results'):
    plt.figure()
    # join the different processed abstracts together
    long_string = ','.join(x)
    # world cloud analysis with visualization
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color="steelblue")
    wordcloud.generate(long_string)
    wordcloud.to_image().save(output_path)
    # wordcloud.to_image().show()
    return


def plot_3d_graph(x=np.array([]), y=np.array([]), z=np.array([]), xlabel='x', ylabel='y', zlabel='z', title="coherence graph", k=0.0, output_base_path='../results'):
    plt.figure()
    ax = plt.axes(projection='3d')
    # c = x + y
    ax.scatter(x, y, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.savefig("%s/coherence_graph_fixed_k%s.png" % (output_base_path, k))
    # plt.show()
    return


def plot_2d_graph(x=[], y=[], xlabel='x', ylabel='y', title='coherence graph', alpha="auto", beta="auto", output_base_path='../results'):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend('coherence_values', loc='best')
    plt.title(title)
    plt.savefig("%s/coherence_graph_fixed_a%s_b%s.png" % (output_base_path, alpha, beta))
    # plt.show()
    return


def tsne_plot(element, n_components=2, perplexity=25, early_exaggeration=12, learning_rate=0.51, n_iter=1000, init='pca', method='barnes_hut', random_state=0, k=8, alpha='auto', beta='auto', output_base_path='../results'):
    plt.figure()
    docs = list(map(lambda e: e['topics'], element))
    x = np.array(docs)
    # x = np.array([list_to_array(doc) for doc in docs])  # use this if minimum_probability of the lda model is > 0.0
    y = [0] * len(docs)  # choose the label for each doc     # TODO implement the colors

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
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title="LDA docs T-SNE projection")
    plt.savefig("%s/tsne_plot_k%s_a%s_b%s.png" % (output_base_path, k, alpha, beta))
    # plt.show()
    return


def get_best_lda_model(lda_models_stats):
    coherence_values = list(map(lambda x: x['coherence'], lda_models_stats))
    index_max = coherence_values.index(min(coherence_values))  # max when using 'c_v' coherence measure, min when using 'u_mass'(?)
    return lda_models_stats[index_max]


def list_to_array(x, k):
    res = [0] * k
    for i in x:
        res[i[0]] = i[1]
    return res


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True is to remove punctuations


def lemmatization(texts, allowed_postags=['VERB', 'NOUN', 'ADJ', 'ADV', 'PROPN', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PRON', 'PUNCT', 'SCONJ', 'X', 'SYM']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def compute_ldamodel(corpus, data, dictionary, k, alpha, beta, coherence_measure, random_state=100, chunksize=2000, passes=10, iterations=50):
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=k,
                         random_state=random_state,
                         chunksize=chunksize,
                         passes=passes,
                         alpha=alpha,
                         eta=beta,
                         minimum_probability=0.0,
                         iterations=iterations,
                         per_word_topics=True
                         )
    coherence_model = CoherenceModel(model=lda_model,
                                     corpus=corpus,
                                     texts=data,
                                     dictionary=dictionary,
                                     coherence=coherence_measure
                                     )
    return lda_model, coherence_model.get_coherence()


if __name__ == '__main__':
    freeze_support()

    # 0. INITIALIZE VARIABLES
    # spark variables
    conf = SparkConf().setAppName('AMiner-WhoIsWho Stats').setMaster('local[*]')
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sqlContext = SQLContext(sc)
    # stopwords initialization
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    # stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    stop_words.extend(['result', 'effect', 'model', 'approach', 'analysis', 'study', 'activity', 'development', 'method', 'system'])
    # initialize spacy 'en' model, keeping only tagger component (for efficiency)
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])  # spacy pipeline: tokenizer, lemmatizer, tagger

    # enable/disable steps
    load_dictionary = False
    hyperparameters_tuning = False
    overwrite_model = True
    generate_lda_vis = True
    generate_lda_docs = True
    corpus_generation = "tfidf"  # allowed values: ['bow', 'tfidf']
    ontology_dictionary = False

    # directories
    output_base_path = "../results"

    # input dataset
    input_path = "../datasets/Aminer-WhoIsWho (na-v3)/processed/pubs_enriched.json"
    text_field = "abstract"

    # baseline parameters for the lda model
    baseline_k = 4
    baseline_alpha = "auto"
    baseline_beta = "auto"
    passes = 20  # epochs
    chunksize = 2000

    # 1. LOADING DATA
    # read data into papers
    papers = sc.textFile(input_path)\
        .map(json.loads)\
        .filter(lambda x: x[text_field])\
        .toDF()\
        .toPandas()
    # print(papers.head())

    # 2. DATA CLEANING
    # remove useless columns
    papers = papers.drop(columns=['keywords', 'authors', 'venue', 'year', 'aids'], axis=1)
    # print(papers.head())

    # remove punctuation and convert to lowercase
    papers["%s_processed" % text_field] = \
        papers[text_field]\
        .map(lambda x: re.sub('[,\.!?%$()0123456789:;]', '', x))  # punctuations
    papers['%s_processed' % text_field] = \
        papers['%s_processed' % text_field]\
        .map(lambda x: x.lower())  # lowercase
    # print(papers['%s_processed' % text_field].head())

    # 3. EXPLORATORY ANALYSIS
    # plot_wordcloud(x=list(papers['%s_processed' % text_field].values), output_path='%s/worldcloud_before.png' % output_base_path)

    # 4. PREPARING DATA FOR LDA ANALYSIS
    # create the list of words for the vocabulary
    data = papers['%s_processed' % text_field].values.tolist()  # collect paper texts
    data_words = list(sent_to_words(data))  # collect paper words
    # print(data_words[:1][0][:30])

    # 5. PHRASE MODELING
    # create bigrams
    bigram = models.Phrases(data_words, min_count=5, threshold=100)  # higher parameters' values, harder for words to be combined
    bigram_mod = models.phrases.Phraser(bigram)
    data_words_nostops = remove_stopwords(data_words)  # remove stopwords
    # data cleaning
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]  # form bigrams
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'PROPN', 'X'])

    # 6. DATA TRANSFORMATION
    # create dictionary
    id2word = []
    if load_dictionary:
        id2word = corpora.Dictionary.load("../results/dictionary_no_below5_no_above0.7")
    else:
        if ontology_dictionary:
            print("Ontology dictionary NOT implemented")
            exit()
        else:
            id2word = corpora.Dictionary(data_lemmatized)
            id2word.filter_extremes(no_below=5, no_above=0.7)
            id2word.save("../results/dictionary_no_below5_no_above0.7")

    # TODO use ontology to create dictionary

    texts = data_lemmatized  # create corpus

    plot_wordcloud(x=list(map(lambda t: ' '.join(t), texts)), output_path='%s/wordcloud_after.png' % output_base_path)

    corpus = [id2word.doc2bow(text) for text in texts]  # compute term document frequency
    if corpus_generation == 'tfidf':
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]
    print("Corpus size: %s" % len(corpus))
    # print(corpus[:1]) #word_id, word frequency

    # 7. HYPERPARAMETERS TUNING (parameters to be tuned by the scientist)
    best_k = baseline_k
    best_alpha = baseline_alpha
    best_beta = baseline_beta
    lda_model = []
    if hyperparameters_tuning:
        pbar = tqdm.tqdm(total=9)  # initiate progress bar

        # 1. alpha and beta fixed, varying num_topics
        topics_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        lda_models_stats = []
        for k in topics_range:
            lda_model, cv = compute_ldamodel(corpus=corpus,
                                             data=data_lemmatized,
                                             dictionary=id2word,
                                             k=k,
                                             alpha=baseline_alpha,
                                             beta=baseline_beta,
                                             coherence_measure='u_mass',
                                             passes=passes,
                                             chunksize=chunksize
                                             )
            # termite could be used to evaluate topics models
            lda_models_stats.append(dict(model=lda_model, k=k, coherence=cv))
            pbar.update(1)

        plot_2d_graph(x=list(map(lambda x: x['k'], lda_models_stats)),
                      y=list(map(lambda x: x['coherence'], lda_models_stats)),
                      xlabel='Number of Topics (k)',
                      ylabel='Coherence Score',
                      title='Coherence Graph with Alpha=%s and Beta=%s' % (baseline_alpha, baseline_beta),
                      alpha=str(baseline_alpha),
                      beta=str(baseline_beta),
                      output_base_path=output_base_path
                      )
        with open("%s/coherence_stats_a%s_b%s.csv" % (output_base_path, baseline_alpha, baseline_beta), 'w') as f:  # create the output html
            writer = csv.writer(f)
            writer.writerow(["k", "coherence"])
            writer.writerows([[x['k'], x['coherence']] for x in lda_models_stats])

        # 2. pick the best number of topics (higher coherence)
        best_lda_model_stats = get_best_lda_model(lda_models_stats)
        best_k = best_lda_model_stats['k']
        print("The optimal number of topics (k) is: %s" % best_k)
        lda_model = best_lda_model_stats['model']
        lda_model.save("%s/lda_model_k%s_a%s_b%s" % (output_base_path, best_k, best_alpha, best_beta))

        pbar.close()
    else:
        # 8. TRAIN THE FINAL MODEL USING THE ABOVE SELECTED PARAMETERS
        if overwrite_model:
            lda_model = LdaModel(corpus=corpus,
                                 id2word=id2word,
                                 num_topics=baseline_k,
                                 random_state=100,
                                 chunksize=100,
                                 passes=10,
                                 alpha=baseline_alpha,
                                 eta=baseline_beta,
                                 minimum_probability=0.0
                                 )
            lda_model.save("%s/lda_model_k%s_a%s_b%s" % (output_base_path, baseline_k, baseline_alpha, baseline_beta))
        else:
            lda_model = LdaModel.load("%s/lda_model_k%s_a%s_b%s" % (output_base_path, baseline_k, baseline_alpha, baseline_beta))

    print("Generated Model with K=%s Alpha=%s Beta=%s" % (best_k, best_alpha, best_beta))

    # 9. ANALYZING LDA MODEL RESULTS
    if generate_lda_vis:
        LDAvis_data_filepath = os.path.join('%s/ldavis_prepared_k%s_a%s_b%s' % (output_base_path, best_k, best_alpha, best_beta))

        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:  # create the output html
            pickle.dump(LDAvis_prepared, f)
        with open(LDAvis_data_filepath, 'rb') as f:  # load the output html
            LDAvis_prepared = pickle.load(f)

        pyLDAvis.save_html(LDAvis_prepared, '%s/ldavis_prepared_k%s_a%s_b%s.html' % (output_base_path, best_k, best_alpha, best_beta))

    # 10. CONVERT CORPUS TO VECTOR
    if generate_lda_docs:
        lda_docs = lda_model[corpus]  # compute the distribution of topics in each document
        papers = [dict(id=papers['id'].iloc[i], topics=list_to_array(lda_docs[i][0], best_k)) for i in range(0, len(lda_docs))]
        with open("%s/lda_docs_k%s_a%s_b%s" % (output_base_path, best_k, best_alpha, best_beta), "wb") as fp:
            pickle.dump(papers, fp)
    else:
        with open("%s/lda_docs_k%s_a%s_b%s", "rb") as fp:
            papers = pickle.load(fp)

    # 11. VISUALIZE THE RESULT WITH TSNE
    tsne_plot(element=papers, k=best_k, alpha=str(best_alpha), beta=str(best_beta), output_base_path=output_base_path)
