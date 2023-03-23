# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, IDF, CountVectorizerModel
from pyspark.ml.clustering import LDA, DistributedLDAModel
import csv
from pyspark.ml.functions import vector_to_array
from plot_utils import *
from pyspark.sql.functions import monotonically_increasing_id

# initializations
spark = sparknlp.start()


def prepare_data(input_data, input_field, output_data_path="datasets/data_tokens.json", save=False):
    """
    Prepare the data for LDA: transform plain text into clean tokens
    Arguments:
        input_data: the input data (plain text to be analyzed)
        output_data_path: location of the output file
    Returns:
        data_tokens: the data transformed into tokens (id, tokens)
    """
    # convert input dataframe to document (required by spark nlp)
    document_assembler = DocumentAssembler()\
        .setInputCol(input_field)\
        .setOutputCol("document")\
        .setCleanupMode("shrink")
    # split input field to tokens
    tokenizer = Tokenizer()\
        .setInputCols(["document"])\
        .setOutputCol("token")
    # clean unwanted characters and garbage
    normalizer = Normalizer()\
        .setInputCols(["token"])\
        .setOutputCol("normalized")
    # remove stopwords
    stopwords_cleaner = StopWordsCleaner()\
        .setInputCols("normalized")\
        .setOutputCol("cleanTokens")\
        .setCaseSensitive(False)
    # stem the words to bring them to the root form.
    stemmer = Stemmer()\
        .setInputCols(["cleanTokens"])\
        .setOutputCol("stem")
    # finish the structure to be ready for lda
    finisher = Finisher()\
        .setInputCols(["stem"])\
        .setOutputCols(["tokens"])\
        .setOutputAsArray(True)\
        .setCleanAnnotations(False)

    nlp_pipeline = Pipeline(
        stages=[document_assembler,
                tokenizer,
                normalizer,
                stopwords_cleaner,
                stemmer,
                finisher])

    # train the pipeline
    nlp_model = nlp_pipeline.fit(input_data)

    # apply pipeline to transform dataframe
    data_processed = nlp_model.transform(input_data)

    # select columns we need
    data_tokens = data_processed.select("id", "tokens")

    if save:
        data_tokens.write.mode('overwrite').json(output_data_path)

    return data_tokens


def create_vocabulary(data_tokens, output_path="results/vocabulary", maxDF=0.6, minDF=100, minTF=1, vocabSize=1<<18, save=False):
    """
    Create the vocabulary and saves it.
    Arguments:
        data_tokens: the tokenized input data
        output_path: location of the vocabulary
        maxDF: max number (could be percentage) of different documents a term could appear in to be included in the vocabulary
        minDF: min number (could be percentage) of different documents a term must appear in to be included in the vocabulary
        minTF: min number (could be percentage) the same term must appear in a single document to be included in the vocabulary (protect from rare terms in a document)
        vocabSize: max size of the vocabulary (number of terms)
    Returns:
        cv_model: the count vectorizer model
    """
    cv = CountVectorizer(inputCol="tokens",
                         outputCol="termFrequencies",
                         maxDF=maxDF,
                         minDF=minDF,
                         minTF=minTF,
                         vocabSize=vocabSize)

    cv_model = cv.fit(data_tokens)

    if save:
        cv_model.write().overwrite().save(output_path)

    return cv_model


def create_corpus(data_tokens, vocabulary, output_data_path="datasets/processed/corpus", save=False):
    """
    Create corpus of the data.
    Arguments:
        data_tokens: the tokenized input data
        vocabulary: the vocabulary to be used for the transformation
        output_data_path: location of the output data (corpus)
    Returns:
        corpus: the corpus
    """
    data_tf = vocabulary.transform(data_tokens)

    idf = IDF(inputCol="termFrequencies", outputCol="features")  # TODO literature says to feed LDA with TF, not IDF
    idf_model = idf.fit(data_tf)

    corpus = idf_model.transform(data_tf).select("id", "features")

    if save:
        corpus.write.mode('overwrite').save(output_data_path)

    return corpus


def compute_lda_model(k, corpus, output_data_path, save=False):
    """
    Creates and saves the LDA model.
    Arguments:
        k: number of topics
        corpus: the corpus
        output_data_path: location for the LDA model
    Returns:
        model: the LDA model
        lp: the perplexity score
    """
    lda = LDA(k=k, maxIter=50, optimizer="em")
    model = lda.fit(corpus)
    lp = model.logPerplexity(corpus)

    if save:
        model.write().overwrite().save("%s/lda_model_k%s" % (output_data_path, k))

    return model, lp


def lda_hyperparameters_tuning(corpus, output_data_path, topics_range=[2, 3, 4, 5, 6, 7, 8, 9, 10], plots_path="results", save=False):
    """
    Tunes the hyperparameters of the model.
    Arguments:
        corpus: the corpus (data processed by the vocabulary)
        output_data_path: location of the trained models output
        topics_range: range of topics to test
        plots_path: location for the plots
    Returns:
        lda_model: the best lda model in terms of best perplexity score
    """
    # initializations
    best_k = 100
    best_lp = 100
    best_lda_model = []

    lda_models_stats = []
    for k in topics_range:
        lda_model, lp = compute_lda_model(k=k, corpus=corpus, output_data_path=output_data_path, save=save)
        lda_models_stats.append(dict(model=lda_model, k=k, perplexity=lp))
        if lp < best_lp:
            best_lp = lp
            best_k = k
            best_lda_model = lda_model

    plot_2d_graph(x=list(map(lambda x: x['k'], lda_models_stats)),
                  y=list(map(lambda x: x['perplexity'], lda_models_stats)),
                  xlabel='Number of Topics (k)',
                  ylabel='Perplexity',
                  title='Perplexity graph varying k',
                  output_base_path=plots_path
                  )
    with open("%s/perplexity_stats.csv" % (plots_path), 'w') as f:  # create the output html
        f.truncate(0)
        writer = csv.writer(f)
        writer.writerow(["k", "perplexity"])
        writer.writerows([[x['k'], x['perplexity']] for x in lda_models_stats])

    print("The optimal number of topics (k) is: %s" % best_k)
    return best_lda_model


def generate_lda_docs(corpus, lda_model, output_data_path="results/lda_topics", save=False):
    """
    Generates the topic vectors for the given corpus.
    Arguments:
        corpus: the corpus (the data transformed by the vocabulary)
        lda_model: the LDA model to be used
        output_data_path: location of the output (the data vectorized)
    Returns:
         topics: the vectorized data
    """
    topics = lda_model.transform(corpus).withColumn("topicsVector", vector_to_array("topicDistribution")).select("id", "topicsVector")

    if save:
        topics.write.mode('overwrite').json(output_data_path)

    return topics


def lda_inference(input_data_path, input_field, output_data_path, lda_model_path, vocabulary_path, save=True):
    """
    Topic inference with LDA for the input data
    Arguments:
        input_data_path: location of the input collection
        input_field: field to be processed
        output_data_path: location of the output collection
        lda_model_path: location of the LDA model to be used for the inference
        vocabulary_path: location of the vocabulary to be used for the transformation
    Returns:
        lda_topics: the data with inferred topics
    """
    publications = spark.read.json(input_data_path)
    publications_tokens = prepare_data(input_data=publications,
                                       input_field=input_field)
    vocabulary = CountVectorizerModel.load(vocabulary_path)
    corpus = create_corpus(data_tokens=publications_tokens,
                           vocabulary=vocabulary)
    lda_model = DistributedLDAModel.load(lda_model_path)

    lda_topics = generate_lda_docs(corpus=corpus,
                                   lda_model=lda_model,
                                   output_data_path=output_data_path,
                                   save=save)
    return lda_topics


def lda_analysis(input_data_path, input_field, data_tokens_path, vocabulary_path, corpus_path, lda_models_base_path, lda_vis_path, maxDF=0.7, minDF=5, minTF=1, vocabSize=1<<18, load_vocabulary=False, load_corpus=False, load_tokens=False, save=True):
    """
    Performs the LDA analysis to chose the best number of topics (k)
    Arguments:
        input_data_path: location of the input JSON data
        input_field: the field to be used for the analysis
        data_tokens_path: location of the tokenized JSON data
        vocabulary_path: location of the vocabulary
        corpus_path: location of the corpus data (transformed by the vocabulary)
        lda_models_base_path: base location for the created LDA models
        lda_vis_path: location of the HTML file for the LDA visualization
        maxDF: max number (could be percentage) of different documents a term could appear in to be included in the vocabulary
        minDF: min number (could be percentage) of different documents a term must appear in to be included in the vocabulary
        minTF: min number (could be percentage) the same term must appear in a single document to be included in the vocabulary (protect from rare terms in a document)
        vocabSize: max size of the vocabulary (number of terms)
        load_vocabulary: load an existing vocabulary
        load_corpus: load an existing corpus
        load_tokens: load existing tokens
    Returns:
        lda_model: the best lda model in terms of perplexity score
    """
    if load_tokens:
        data_tokens = spark.read.json(data_tokens_path)
    else:
        data = spark.read.json(input_data_path)
        data_tokens = prepare_data(input_data=data,
                                   input_field=input_field,
                                   output_data_path=data_tokens_path,
                                   save=save)

    if load_vocabulary:
        vocabulary = CountVectorizerModel.load(vocabulary_path)
    else:
        vocabulary = create_vocabulary(data_tokens=data_tokens,
                                       output_path=vocabulary_path,
                                       maxDF=maxDF,
                                       minDF=minDF,
                                       minTF=minTF,
                                       vocabSize=vocabSize,
                                       save=save)
    if load_corpus:
        corpus = spark.read.load(corpus_path)
    else:
        corpus = create_corpus(data_tokens=data_tokens,
                               vocabulary=vocabulary,
                               output_data_path=corpus_path,
                               save=save)

    lda_model = lda_hyperparameters_tuning(corpus=corpus,
                                           output_data_path=lda_models_base_path,
                                           save=save)

    lda_topics = generate_lda_docs(corpus=corpus,
                                   lda_model=lda_model)

    generate_lda_visualization(data_tokens=data_tokens,
                               vocabulary=vocabulary,
                               lda_topics=lda_topics,
                               lda_model=lda_model,
                               html_output_file=lda_vis_path)

    return lda_model


def create_custom_vocabulary(input_data_path, output_path, maxDF, minDF, minTF=1, save=True):
    """
    Create a custom vocabulary from a list of strings.
    Arguments:
        input_data_path: location of the input raw file (list of subjects)
        output_path: location for the vocabulary
    Returns:
        vocabulary: the vocabulary
    """
    raw_data = spark.sparkContext.textFile(input_data_path).map(eval)\
        .map(lambda x: dict(subjects=' '.join(x)))\
        .toDF().withColumn("id", monotonically_increasing_id())

    data_tokens = prepare_data(input_data=raw_data, input_field="subjects")

    vocabulary = create_vocabulary(data_tokens=data_tokens,
                                   output_path=output_path,
                                   maxDF=maxDF,
                                   minDF=minDF,
                                   minTF=minTF,
                                   save=save)

    return vocabulary


if __name__ == '__main__':
    vocabulary = create_custom_vocabulary(input_data_path="datasets/openaire_subjects",
                                          output_path="results/openaire_subjects_vocabulary",
                                          maxDF=0.6,
                                          minDF=1.0,
                                          save=False)

    lda_analysis(input_data_path="datasets/processed/aminer_wiw_pubs.json",
                 input_field="abstract",
                 data_tokens_path="datasets/processed/aminer_wiw_tokens.json",
                 vocabulary_path="results/openaire_subjects_vocabulary",
                 corpus_path="datasets/processed/aminer_wiw_pubs_corpus_oa_subjects",
                 lda_models_base_path="outputs/lda_models_oa_subjects",
                 lda_vis_path="results/lda_vis_test.html",
                 load_vocabulary=True)

    lda_inference(input_data_path="datasets/processed/aminer_wiw_pubs.json",
                  output_data_path="results/lda_topics_test",
                  input_field="abstract",
                  lda_model_path="outputs/lda_models/lda_model_k2",
                  vocabulary_path="results/vocab_test",
                  save=True)
