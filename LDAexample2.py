#LDA: Latent Dirichlet Allocation
#PARAMETERS OF LDA:
#alpha: document-topic density, high alpha=documents made up of more topics, more specific topic distribution per document
#beta: topic-word density, high beta=topics made up most of the words, more specific word distribution per topic
import os
import pickle
import re
from multiprocessing import freeze_support

import nltk
import numpy as np
import pandas as pd
import pyLDAvis
import seaborn as sns
import spacy
import tqdm
from gensim import models, corpora
from gensim.models import CoherenceModel
from gensim.models import LdaMulticore
from gensim.utils import ClippedCorpus
from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from pyLDAvis import gensim
from sklearn.manifold import TSNE
from wordcloud import WordCloud

#stopwords initialization
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# initialize spacy 'en' model, keeping only tagger component (for efficiency)
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

if __name__ == '__main__':
    freeze_support()

    #1. LOADING DATA
    #read data into papers
    papers = pd.read_csv("../datasets/NIPS Papers/papers.csv", nrows=100) # .sample(100) #to reduce the number of inputs (test purpose)
    # print(papers.head())

    #2. DATA CLEANING
    #remove useless columns
    papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1)
    # print(papers.head())

    #remove punctuation and convert to lowercase
    papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]','', x)) #punctuations
    papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower()) #lowercase
    # print(papers['paper_text_processed'].head())

    #3. EXPLORATORY ANALYSIS
    #join the different processed titles together
    long_string = ','.join(list(papers['paper_text_processed'].values)) #put all the words of the papers into a unique string

    #world cloud analysis with visualization
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color="steelblue")
    wordcloud.generate(long_string)
    wordcloud.to_image().show()

    #4. PREPARING DATA FOR LDA ANALYSIS
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(simple_preprocess(str(sentence), deacc=True)) #deacc=True is to remove punctuations

    #create the list of words for the vocabulary
    data = papers.paper_text_processed.values.tolist() #collect paper texts
    data_words = list(sent_to_words(data)) #collect paper words

    # print(data_words[:1][0][:30])

    #5. PHRASE MODELING: BIGRAMS AND TRIGRAMS
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

    #build the bigram and trigram models
    bigram = models.Phrases(data_words, min_count=5, threshold=100) #higher parameters' values, harder for words to be combined
    trigram = models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    #data cleaning
    data_words_nostops = remove_stopwords(data_words) #remove stopwords
    data_words_bigrams = make_bigrams(data_words_nostops) #form bigrams
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) #do lemmatization keeping only noun, adj, vb, adv

    # print(data_lemmatized[:1])

    #6. DATA TRANSFORMATION
    id2word = corpora.Dictionary(data_lemmatized) #create dictionary #TODO insert the dictionary: test with list of words (topics)
    texts = data_words #create corpus (the guide says data_lemmatized, it should be wrong)
    corpus = [id2word.doc2bow(text) for text in texts] #compute term document frequency

    # print(corpus[:1]) #word_id, word frequency

    # #7. BASELINE LDA MODEL TRAINING
    # num_topics=10 #alpha and beta are set prior to 1/num_topics (default)
    # #build LDA model
    # lda_model = LdaMulticore(corpus=corpus,
    #                          id2word=id2word,
    #                          num_topics=num_topics,
    #                          random_state=100,
    #                          chunksize=100, #how many documents are processed at a time (higher values speed up training)
    #                          passes=10, #epochs
    #                          per_word_topics=True
    #                          )
    #
    # # pprint(lda_model.print_topics())
    # # doc_lda = lda_model[corpus] #convert documents to lda
    #
    # #compute the coherence of the baseline model
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence="c_v")
    # baseline_coherence_lda = coherence_model_lda.get_coherence()
    #
    # # print('\nCoherence Score: ', baseline_coherence_lda)
    #
    # #8. HYPERPARAMETERS TUNING (parameters to be tuned by the scientist)
    # def compute_coherence_values(corpus, dictionary, k, a, b):
    #     lda_model = LdaMulticore(corpus=corpus,
    #                              id2word=dictionary,
    #                              num_topics=k,
    #                              random_state=100,
    #                              chunksize=100,
    #                              passes=10,
    #                              alpha=a,
    #                              eta=b
    #                              )
    #     coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    #     return coherence_model_lda.get_coherence()
    #
    # #prepare data for the hyperparameters tuning
    # grid = {}
    # grid['Validation_Set'] = {}
    # #topics range
    # min_topics = 2
    # max_topics = 11
    # step_size = 1
    # topics_range = range(min_topics, max_topics, step_size)
    # #alpha parameter
    # alpha = list(np.arange(0.01, 1, 0.3))
    # alpha.append('symmetric')
    # alpha.append('asymmetric')
    # #beta parameter
    # beta = list(np.arange(0.01, 1, 0.3))
    # beta.append('symmetric')
    # #validation sets
    # num_of_docs = len(corpus)
    # corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
    #                # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
    #                ClippedCorpus(corpus, int(num_of_docs*0.75))]
    #
    # corpus_title = ['75% Corpus', '100% Corpus']
    #
    # model_results = {'Validation_Set': [],
    #                  'Topics': [],
    #                  'Alpha': [],
    #                  'Beta': [],
    #                  'Coherence': []
    #                 }
    #
    # if 1==1:
    #     pbar = tqdm.tqdm(total=540) #initiate progress bar
    #     for i in range(len(corpus_sets)): #iterate through validation corpuses
    #         for k in topics_range: #iterate over number of topics
    #             for a in alpha: #iterate over alpha values
    #                 for b in beta: #iterate over beta values
    #                     #get the coherence score for the given parameters
    #                     cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, k=k, a=a, b=b)
    #                     #save the model results
    #                     model_results['Validation_Set'].append(corpus_title[i])
    #                     model_results['Topics'].append(k)
    #                     model_results['Alpha'].append(a)
    #                     model_results['Beta'].append(b)
    #                     model_results['Coherence'].append(cv)
    #                     pbar.update(1)
    #     pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    #     pbar.close()
    #
    # #how to choose the best hyperparameters:
    # #the higher is the coherence score the better is
    # #1. fix alpha and beta and choose the num_topic with the best coherence
    # #2. select highest coherence scores for num_topics=8 and pick alpha and beta
    # #3. see the difference with the baseline coherence score

    #9. TRAIN THE FINAL MODEL USING THE ABOVE SELECTED PARAMETERS
    num_topics = 8
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=id2word,
                             num_topics=num_topics,
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             alpha=0.01,
                             eta=0.9,
                             workers=1)

    #10. ANALYZING LDA MODEL RESULTS

    # pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('../results/ldavis_prepared')

    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f: #create the output html
        pickle.dump(LDAvis_prepared, f)
    with open(LDAvis_data_filepath, 'rb') as f: #load the output html
            LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, '../results/ldavis_prepared.html')

    #11. CONVERT CORPUS TO VECTOR
    lda_docs = lda_model[corpus]

    print(len(corpus))

    #printing the topic associations with the documents
    count = 0
    for i in lda_docs:
        print("doc : ", count, i)
        count += 1

    # import pickle
    # with open("lda_docs", "wb") as fp:
    #     pickle.dump(lda_docs, fp)
    # with open("lda_docs", "rb") as fp:
    #     lda_docs = pickle.load(fp)

    #12. VISUALIZE THE RESULT WITH TSNE

    def listToArray(topicDistribution): #convert the lda doc (list of topics) to array
        res = [0] * num_topics
        for topic in topicDistribution:
            res[topic[0]] = topic[1]
        return res

    def tsne_plot(docs, n_components=2, perplexity=25, early_exaggeration=12, learning_rate=0.51, n_iter=1000, init='pca', method='barnes_hut', random_state=0):

        x = np.array([listToArray(doc) for doc in docs])
        y = [0] * len(docs) # choose the label for each doc

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
        plt.show()
        return

    tsne_plot(docs=lda_docs)

