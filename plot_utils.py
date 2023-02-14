from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from pyspark.sql.functions import udf, col, size, explode, regexp_replace, trim, lower, lit
import pyLDAvis


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


def plot_2d_graph(x=[], y=[], xlabel='x', ylabel='y', title='coherence graph', output_base_path='../results'):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend('coherence_values', loc='best')
    plt.title(title)
    plt.savefig("%s/%s.png" % (output_base_path, title))
    # plt.show()
    return


def tsne_plot(element, n_components=2, perplexity=25, early_exaggeration=12, learning_rate=0.51, n_iter=1000, init='pca', method='barnes_hut', random_state=0, output_path='../results/tsne_plot.png'):
    plt.figure()
    docs = list(map(lambda e: e['topics'], element))
    x = np.array(docs)
    y = list(map(lambda e: e['gt_id'], element))

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
    plt.savefig(output_path)
    # plt.show()
    return


def format_data_to_pyldavis(df_filtered, count_vectorizer, transformed, lda_model):
    xxx = df_filtered.select((explode(df_filtered.tokens)).alias("words")).groupby("words").count()
    word_counts = {r['words']:r['count'] for r in xxx.collect()}
    word_counts = [word_counts[w] for w in count_vectorizer.vocabulary]

    data = {'topic_term_dists': np.array([row for row in lda_model.describeTopics(maxTermsPerTopic=len(count_vectorizer.vocabulary)).select(col('termWeights')).toPandas()['termWeights']]),
            'doc_topic_dists': np.array([x for x in transformed.select(["topicsVector"]).toPandas()['topicsVector']]),
            'doc_lengths': [r[0] for r in df_filtered.select(size(df_filtered.tokens)).collect()],
            'vocab': count_vectorizer.vocabulary,
            'term_frequency': word_counts}

    return data


def filter_bad_docs(data):
    # this is, because for some reason some docs apears with 0 value in all the vectors, or the norm is not 1, so I filter those docs.
    bad = 0
    doc_topic_dists_filtered = []
    doc_lengths_filtered = []

    for x,y in zip(data['doc_topic_dists'], data['doc_lengths']):
        if np.sum(x)==0:
            bad+=1
        elif np.sum(x) != 1:
            bad+=1
        elif np.isnan(x).any():
            bad+=1
        else:
            doc_topic_dists_filtered.append(x)
            doc_lengths_filtered.append(y)

    data['doc_topic_dists'] = doc_topic_dists_filtered
    data['doc_lengths'] = doc_lengths_filtered


def generate_lda_visualization(data_tokens, vocabulary, lda_topics, lda_model, html_output_file):
    """
    Generate the HTML file to visualize the LDA model results.
    Arguments:
        data_tokens: the tokenized data
        vocabulary: the vocabulary
        lda_topics: the vectorized data (processed with the vocabulary)
        lda_model: the model to be used
        html_output_file: the output file
    """
    data = format_data_to_pyldavis(data_tokens, vocabulary, lda_topics, lda_model)
    filter_bad_docs(data)
    py_lda_prepared_data = pyLDAvis.prepare(**data)
    pyLDAvis.save_html(py_lda_prepared_data, html_output_file)
