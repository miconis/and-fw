from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from sklearn.manifold import TSNE


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


def tsne_plot(element, n_components=2, perplexity=25, early_exaggeration=12, learning_rate=0.51, n_iter=1000, init='pca', method='barnes_hut', random_state=0, output_path='../results/tsne_plot.png'):
    plt.figure()
    docs = list(map(lambda e: e['topics'], element))
    x = np.array(docs)
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
    plt.savefig(output_path)
    # plt.show()
    return
