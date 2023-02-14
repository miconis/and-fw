from plot_utils import *
from utility import *
from pyspark import SparkConf
from pyspark import SparkContext

def authors_flat_map(x):
    """
    Create list of authors from the publication.
    Arguments:
        x: (id, (publication, topics))
    Returns:
        authors: the list of authors
    """
    authors = []
    topics_vector = x[1][1]['topics']
    publication = x[1][0]
    for author in publication['authors']:
        coauthors = publication['authors'].copy()
        coauthors.remove(author)
        authors.append(dict(id=md5(author['name'] + publication['id']),
                            name=author['name'],
                            org=author['org'],
                            pub_id=publication['id'],
                            gt_id=author.get('id', ''),
                            keywords=publication['keywords'],
                            venue=publication['venue'],
                            year=publication['year'],
                            topics=topics_vector,
                            coauthors=coauthors
                            ))
    return authors


def author_extractor(sc, input_data_path, lda_topics_path, output_data_path):
    """
    Extract author from input publications and create dataset.
    Arguments:
        sc: the spark context
        input_data_path: location of the json input file (the publication data)
        lda_topics_path: location of the json input file (the topics data)
        output_data_path: location of the json output file
    Returns:
        data_lemmatized: the lemmatization of the data in the form [(id, lemmatization)]
    """
    data = sc.textFile(input_data_path).map(json.loads).map(lambda x: (x['id'], x))
    lda_topics = sc.textFile(lda_topics_path).map(json.loads).map(lambda x: (x['id'], x))

    authors = data.join(lda_topics).flatMap(authors_flat_map)

    print("Author extracted: %s" % authors.count())
    list_to_file(authors.collect(), output_data_path)

    return authors


if __name__ == '__main__':
    conf = SparkConf().setAppName('author extractor').setMaster('local[*]')
    sc = SparkContext(conf=conf)

    # author_extractor(sc=sc,
    #                  input_data_path="datasets/processed/aminer_wiw_pubs.json",
    #                  lda_topics_path="datasets/processed/aminer_wiw_pubs_lda_topics.json",
    #                  output_data_path="datasets/processed/aminer_wiw_authors.json")
    #
    # conf = SparkConf().setAppName('author extractor').setMaster('local[*]')
    # sc = SparkContext(conf=conf)
    # papers = sc.parallelize(pd.read_json("datasets/processed/aminer_wiw_authors.json", lines=True).to_dict('records'))\
    #     .filter(lambda x: lnfi(x['name']) == 'zzhang')
    # tsne_plot(element=papers.collect(), output_path="results/tsne_plot_author_wiw_hyang")

    data = sc.textFile("datasets/processed/aminer_wiw_authors.json").map(json.loads)
    data_field_stats(data, "org", "string", "results/authors_orgs_stats.txt")
    data_field_stats(data, "keywords", "list", "results/authors_keywords.txt")
