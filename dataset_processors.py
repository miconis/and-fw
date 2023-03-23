# functions to process datasets of each provider: creates publications json files with author ground truth ids
from utility import *
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
import json
from pyspark.sql.functions import col
import networkx as nx
from networkx.readwrite import json_graph
from skbio.diversity.alpha import shannon
from itertools import groupby


def create_json_graph(id, x):
    simrels = list(x)
    graph = dict(nodes=[], edges=[])
    ids = []  # list of node ids

    # variables to compute statistics
    gt_ids = []  # list of ground truth ids
    no_ids = 0  # number of elements with no ground truth id

    for simrel in simrels:
        # check if the node already exists
        if simrel['a'] not in ids:
            ids.append(simrel['a'])
            if simrel['a_id'] != '':
                gt_ids.append(simrel['a_id'])
            else:
                no_ids += 1
            # insert node
            graph["nodes"].append(dict(id=ids.index(simrel['a']), topics=simrel['a_topics'], gt_id=simrel['a_id']))
        if simrel['b'] not in ids:
            ids.append(simrel['b'])
            if simrel['b_id'] != '':
                gt_ids.append(simrel['b_id'])
            else:
                no_ids += 1
            graph["nodes"].append(dict(id=ids.index(simrel['b']), topics=simrel['b_topics'], gt_id=simrel['b_id']))

        # insert edge if does not exist
        edge = (ids.index(simrel['a']), ids.index(simrel['b']))
        if edge not in graph["edges"]:
            graph["edges"].append(edge)

    # shannon-wiener diversity: the input is a list of counters (term 0 appears list[0] times)
    input_list = [len(list(group)) for key, group in groupby(sorted(gt_ids))]
    diversity2 = shannon(input_list)
    input_list.extend([1] * no_ids)  # ground truth ids equal to '' are counted as all different, added separately
    diversity = shannon(input_list)

    return dict(id=id, graph=graph, diversity_only_ids=diversity2, diversity=diversity, no_gt_id=no_ids, nodes_number=len(ids))


def create_json_graph_nx(x):

    simrels = list(x)

    graph = nx.Graph()
    ids = []
    gt_ids = []
    no_ids = 0  # number of elements with no ground truth id

    for simrel in simrels:

        # check if the node already exists
        if simrel['a'] not in ids:
            ids.append(simrel['a'])
            if simrel['a_id'] != '':
                gt_ids.append(simrel['a_id'])
            else:
                no_ids += 1
            # insert node
            graph.add_node(ids.index(simrel['a']), topics=simrel['a_topics'], gt_id=simrel['a_id'])
        if simrel['b'] not in ids:
            ids.append(simrel['b'])
            if simrel['b_id'] != '':
                gt_ids.append(simrel['b_id'])
            else:
                no_ids += 1
            graph.add_node(ids.index(simrel['b']), topics=simrel['b_topics'], gt_id=simrel['b_id'])

        # insert edge if does not exist
        if not graph.has_edge(ids.index(simrel['a']), ids.index(simrel['b'])):
            graph.add_edge(ids.index(simrel['a']), ids.index(simrel['b']))

    # shannon-wiener diversity: the input is a list of counters (term 0 appears list[0] times)
    input_list = [len(list(group)) for key, group in groupby(sorted(gt_ids))]
    diversity2 = shannon(input_list)
    input_list.extend([1] * no_ids)  # ground truth ids equal to '' are counted as all different, added separately
    diversity = shannon(input_list)

    return dict(graph=json_graph.adjacency_data(graph), diversity_only_ids=diversity2, diversity=diversity, no_gt_id=no_ids, nodes_number=len(ids))


def aminer_wiw_to_json(pubs_input_path, authors_gt_path, pubs_output_path):

    # read JSON authors ground truth
    f = open(authors_gt_path)
    authors_gt = json.load(f)

    # JSON publications: un-nest JSON and inject authors' ids
    f = open(pubs_input_path)
    pubs = json.load(f)
    pubs_list = []
    for p_id in pubs:
        pub = pubs[p_id]
        for author in pub['authors']:
            a_key = author['name'].replace("-", "").replace(" ", "_").lower()
            if a_key in authors_gt.keys():
                for a_id in authors_gt[a_key]:
                    if p_id in authors_gt[a_key][a_id]:
                        author['id'] = a_id
                        break
        pubs_list.append(pub)

    print("Number of publications: %s" % len(pubs_list))
    list_to_file(pubs_list, pubs_output_path)


def dedup_result_processor(spark, simrels_path, mergerels_path, authors_path, graphs_path="/tmp/dedup_graphs", save=True):

    simrels = spark.read.load(simrels_path).withColumnRenamed("source", "a").withColumnRenamed("target", "b")
    mergerels = spark.read.load(mergerels_path).withColumnRenamed("source", "group").withColumnRenamed("target", "raw_id")
    authors = spark.read.json(authors_path).select(col("id"), col("name"), col("topics"), col("gt_id"))

    # attach the group id (connected component id) to the simrels
    simrels_with_group = simrels\
        .join(mergerels, simrels.a == mergerels.raw_id, "left")\
        .select(col("a"), col("b"), col("group"))

    # attach topics and ground truth id of the source
    join_res = simrels_with_group\
        .join(authors, simrels_with_group.a == authors.id, "left")\
        .withColumnRenamed("gt_id", "a_id")\
        .withColumnRenamed("topics", "a_topics")\
        .select(col("a"), col("b"), col("group"), col("a_topics"), col("a_id"))

    # attach topics and ground truth id of the target
    join_res = join_res\
        .join(authors, join_res.b == authors.id, "left")\
        .withColumnRenamed("gt_id", "b_id")\
        .withColumnRenamed("topics", "b_topics")\
        .select(col("a"), col("b"), col("group"), col("a_topics"), col("b_topics"), col("a_id"), col("b_id"))

    res = join_res.rdd.map(lambda x: (x['group'], x)).groupByKey().map(lambda x: create_json_graph(x[0], x[1]))

    if save:
        res.map(json.dumps).saveAsTextFile(graphs_path)
    return res


if __name__ == '__main__':
    conf = SparkConf()\
        .setAppName('Dataset Processor')\
        .set("spark.driver.memory", "15g")\
        .setMaster('local[*]')

    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    dedup_result_processor(spark=spark,
                           simrels_path="datasets/dedup/aminer_dedup/aminer_correct_dedup/simrels",
                           mergerels_path="datasets/dedup/aminer_dedup/aminer_correct_dedup/mergerels",
                           authors_path="datasets/processed/aminer_wiw_authors.json",
                           graphs_path="datasets/processed/aminer_correct_dedup_graphs")

    # aminer_wiw_to_json(pubs_input_path="datasets/Aminer-WhoIsWho (na-v3)/train_pub.json",
    #                    authors_gt_path="datasets/Aminer-WhoIsWho (na-v3)/train_author.json",
    #                    pubs_output_path="datasets/processed/aminer_wiw_pubs.json")
