def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pickle
import re
from multiprocessing import freeze_support
import csv

from plot_utils import *
from utility import *

import nltk
import pyLDAvis
import spacy
import tqdm
from gensim import models, corpora
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from pyLDAvis import gensim


def list_to_array(x, k):
    res = [0] * k
    for i in x:
        res[i[0]] = float(i[1])
    return res


def remove_stopwords(texts):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    # stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    stop_words.extend(['result', 'effect', 'model', 'approach', 'analysis', 'study', 'activity', 'development', 'method', 'system'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True is to remove punctuations


def lemmatization(texts, allowed_postags=['VERB', 'NOUN', 'ADJ', 'ADV', 'PROPN', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PRON', 'PUNCT', 'SCONJ', 'X', 'SYM']):
    # initialize spacy 'en' model, keeping only tagger component (for efficiency)
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])  # spacy pipeline: tokenizer, lemmatizer, tagger

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def compute_ldamodel(corpus, data, dictionary, k, alpha, beta, coherence_measure, random_state=100, chunksize=2000, passes=10, iterations=50, output_data_path="results"):
    """
    Create the LDA model.
    Arguments:
        corpus: the corpus of the data
        data: the input data
        dictionary: the dictionary to be used for the model
        k: number of topics
        alpha: document-topic density, high alpha=documents made up of more topics, more specific topic distribution per document
        beta: topic-word density, high beta=topics made up most of the words, more specific word distribution per topic
        coherence_measure: coherence score to be computed to measure the quality of the model
    Returns:
        lda_model: the LDA model
        coherence measure: the coherence of the model
    """
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
    lda_model.save("%s/lda_model_k%s_a%s_b%s" % (output_data_path, k, alpha, beta))
    return lda_model, coherence_model.get_coherence()


def lda_hyperparameters_tuning(corpus_path, data_lemmatized_path, dictionary_path, output_data_path, topics_range = [2, 3, 4, 5, 6, 7, 8, 9, 10], alpha="auto", beta="auto", coherence_measure="u_mass", passes=20, chunksize=2000, plots_path="results"):
    """
    Prepare the data for the LDA algorithm.
    Arguments:
        corpus_path: location of the corpus
        data_lemmatized_path: location of the processed data
        dictionary_path: location of the dictionary
        output_data_path: location of the trained models output
    Returns:
        lda_model: the best lda model in terms of best coherence measure
    """
    pbar = tqdm.tqdm(total=9)  # initiate progress bar
    clear_directory(output_data_path)

    with open(corpus_path, 'rb') as fp:
        corpus = pickle.load(fp)
    data_lemmatized = file_to_list(data_lemmatized_path)
    dictionary = corpora.Dictionary.load(dictionary_path)

    # initializations
    best_k = 100
    best_cv = 100
    best_lda_model = []

    lda_models_stats = []
    for k in topics_range:
        lda_model, cv = compute_ldamodel(corpus=corpus,
                                         data=data_lemmatized,
                                         dictionary=dictionary,
                                         k=k,
                                         alpha=alpha,
                                         beta=beta,
                                         coherence_measure=coherence_measure,
                                         passes=passes,
                                         chunksize=chunksize,
                                         output_data_path=output_data_path
                                         )
        lda_models_stats.append(dict(model=lda_model, k=k, coherence=cv))
        if cv < best_cv:  # switch to > when using 'c_v' coherence measure
            best_cv = cv
            best_k = k
            best_lda_model = lda_model
        pbar.update(1)

    plot_2d_graph(x=list(map(lambda x: x['k'], lda_models_stats)),
                  y=list(map(lambda x: x['coherence'], lda_models_stats)),
                  xlabel='Number of Topics (k)',
                  ylabel='Coherence Score',
                  title='Coherence Graph with Alpha=%s and Beta=%s' % (alpha, beta),
                  alpha=str(alpha),
                  beta=str(beta),
                  output_base_path=plots_path
                  )
    with open("%s/coherence_stats_a%s_b%s.csv" % (plots_path, alpha, beta), 'w') as f:  # create the output html
        f.truncate(0)
        writer = csv.writer(f)
        writer.writerow(["k", "coherence"])
        writer.writerows([[x['k'], x['coherence']] for x in lda_models_stats])

    pbar.close()
    print("The optimal number of topics (k) is: %s" % best_k)
    return best_lda_model


def process_for_lda(input_data_path, output_data_path, input_field='abstract', allowed_postags=['NOUN', 'PROPN', 'X'], plots_path='results'):
    """
    Prepare the data for the LDA algorithm.
    Arguments:
        input_data_path: location of the json input file (the raw data)
        output_data_path: location of the json output file (the data filtered, lemmatized and processed)
        input_field: field of the input data to be processed
        allowed_postags: part of the text to be processed
        plots_path: path of the generated plots
    Returns:
        data_lemmatized: the lemmatization of the data in the form [(id, lemmatization)]
    """
    data = pd.read_json(input_data_path, lines=True)[['id', input_field]]
    # pbar = tqdm.tqdm(total=len(data))  # initiate progress bar

    data["%s_processed" % input_field] = data[input_field].map(lambda x: re.sub('[,\.!?%$()0123456789:;]', '', x))  # remove punctuations
    data['%s_processed' % input_field] = data['%s_processed' % input_field].map(lambda x: x.lower())  # to lowercase

    # plot wordcloud analysis (before)
    plot_wordcloud(x=list(data['%s_processed' % input_field].values), output_path='%s/worldcloud_before.png' % plots_path)

    # create the list of words for the vocabulary
    data_list = data['%s_processed' % input_field].values.tolist()  # collect paper texts
    data_words = list(sent_to_words(data_list))  # collect paper words

    # create bigrams
    bigram = models.Phrases(data_words, min_count=5, threshold=100)  # higher parameters' values, harder for words to be combined
    bigram_mod = models.phrases.Phraser(bigram)
    data_words_nostops = remove_stopwords(data_words)  # remove stopwords
    # data cleaning
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]  # form bigrams
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=allowed_postags)

    list_to_file(data_lemmatized, output_data_path)

    # plot wordcloud analysis (after)
    plot_wordcloud(x=list(map(lambda t: ' '.join(t), data_lemmatized)), output_path='%s/wordcloud_after.png' % plots_path)

    return data_lemmatized


def create_dictionary(input_data_path, no_below=5, no_above=0.7, dictionary_output_path='outputs/dictionary'):
    """
    Create the dictionary for the LDA algorithm.
    Arguments:
        input_data_path: location of the json input file (the data processed)
        no_below: exclude terms in less than...
        no_above: exclude terms in more than...
    Returns:
        dictionary: the dictionary
    """
    data_lemmatized = file_to_list(input_data_path)

    dictionary = corpora.Dictionary(data_lemmatized)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.save(dictionary_output_path)
    return dictionary


def create_corpus(input_data_path, output_data_path, dictionary_path='results/', type='tfidf'):
    """
    Create the corpus for the LDA algorithm.
    Arguments:
        input_data_path: location of the json input file (the data processed)
        output_data_path: location of the corpus file
        dictionary_path: location of the dictionary to be used
        type: type of transformation (tfidf, bow)
    Returns:
        corpus: the corpus
    """
    data_lemmatized = file_to_list(input_data_path)

    dictionary = corpora.Dictionary.load(dictionary_path)
    corpus = [dictionary.doc2bow(text) for text in data_lemmatized]
    if type == 'tfidf':
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]

    with open(output_data_path, "wb") as fp:
        fp.truncate(0)
        pickle.dump(corpus, fp)

    print("Corpus size: %s" % len(corpus))
    return corpus


def generate_lda_vis(lda_data_filepath, lda_model_path, corpus_path, dictionary_path, plots_path="results"):
    """
    Create the corpus for the LDA algorithm.
    Arguments:
        lda_data_filepath: location of the data to be used for the HTML
        lda_model_path: location of the LDA model to be used
        corpus_path: location of the corpus
        dictionary_path: location of the dictionary to be used
        plots_path: location of the plots created by the function
    Returns:
        corpus: the corpus
    """
    LDAvis_data_filepath = os.path.join(lda_data_filepath)

    lda_model = LdaModel.load(lda_model_path)
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    dictionary = corpora.Dictionary.load(dictionary_path)

    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    with open(LDAvis_data_filepath, 'wb') as f:  # create the output html
        pickle.dump(LDAvis_prepared, f)
    with open(LDAvis_data_filepath, 'rb') as f:  # load the output html
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, '%s/ldavis_prepared.html' % plots_path)


def generate_lda_docs(input_data_path, lda_model_path, corpus_path, k, output_data_path):
    """
    Generate the LDA topic vectors.
    Arguments:
        input_data_path: location of the input JSON file
        lda_model_path: location of the LDA model to be used
        corpus_path: location of the corpus to be used
        k: number of topics
        output_data_path: location of the JSON file with topic vectors
    Returns:
        lda_docs: the input with the topic representations
    """
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    lda_model = LdaModel.load(lda_model_path)
    data = pd.read_json(input_data_path, lines=True)

    lda_docs = lda_model[corpus]  # compute the distribution of topics in each document
    data = [dict(id=data['id'].iloc[i], topics=list_to_array(lda_docs[i][0], k)) for i in range(0, len(lda_docs))]

    list_to_file(data, output_data_path)


if __name__ == '__main__':

    freeze_support()

    data_lemmatized = process_for_lda(input_data_path="datasets/processed/aminer_wiw_pubs.json",
                                      output_data_path="datasets/processed/aminer_wiw_pubs_processed.json")

    dictionary = create_dictionary(input_data_path="datasets/processed/aminer_wiw_pubs_processed.json",
                                   dictionary_output_path="outputs/dictionary_aminer_wiw_nobelow5_noabove07")

    corpus = create_corpus(input_data_path="datasets/processed/aminer_wiw_pubs_processed.json",
                           output_data_path="datasets/processed/aminer_wiw_pubs_corpus",
                           dictionary_path="outputs/dictionary_aminer_wiw_nobelow5_noabove07")

    lda_model = lda_hyperparameters_tuning(corpus_path="datasets/processed/aminer_wiw_pubs_corpus",
                                           data_lemmatized_path="datasets/processed/aminer_wiw_pubs_processed.json",
                                           dictionary_path="outputs/dictionary_aminer_wiw_nobelow5_noabove07",
                                           output_data_path="outputs/lda_models")

    generate_lda_vis(lda_data_filepath="results/ldavis_prepared_k10_aauto_bauto",
                     lda_model_path="outputs/lda_models/lda_model_k10_aauto_bauto",
                     corpus_path="datasets/processed/aminer_wiw_pubs_corpus",
                     dictionary_path="outputs/dictionary_aminer_wiw_nobelow5_noabove07")

    generate_lda_docs(input_data_path="datasets/processed/aminer_wiw_pubs.json",
                      lda_model_path="outputs/lda_models/lda_model_k10_aauto_bauto",
                      corpus_path="datasets/processed/aminer_wiw_pubs_corpus",
                      k=10,
                      output_data_path="datasets/processed/aminer_wiw_pubs_lda_topics.json")

    papers = pd.read_json("datasets/processed/aminer_wiw_pubs_lda_topics.json", lines=True).to_dict('records')
    tsne_plot(element=papers, output_path="results/tsne_plot_k10_aauto_bauto")
