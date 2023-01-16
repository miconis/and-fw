# ENRICH PUBLICATION WITH AUTHORS ID TO HAVE A GROUND TRUTH FOR THE CLUSTERING STRATEGY
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
import json
import hashlib
def authorNameToKey(x):
    return x.replace("-", "").replace(" ", "_").lower()

def add(x):
    x[1][0]['aids'] = x[1][1]
    return x

def injectID(x):
    authors = x['authors']
    ids = x['aids']
    for author in authors:
        key = authorNameToKey(author['name'])
        for aid in ids:
            if aid['key'] == key:
                author['aid'] = aid['aid']
            else:
                author['aid'] = ""
    return x

def generateID(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def pubToAuths(pub):
    auths = []
    for author in pub['authors']:
        name = author['name']
        organization = author['org']
        pub_data = dict(pid=pub['id'], title=pub['title'], abstract=pub['abstract'], keywords=pub['keywords'], venue=pub['venue'], year=pub['year'])
        coauthors = []
        for a in pub['authors']:
            if name not in a['name']:
                coauthors.append(a)
        auths.append(dict(id=generateID(name+pub_data['pid']), gtid=author['aid'], name=name, org=organization, publication=pub_data, coauthors=coauthors))
    return auths

conf = SparkConf().setAppName('AMiner-WhoIsWho Stats').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

pubs_path = "../datasets/Aminer-WhoIsWho (na-v3)/train_pub.json"
auths_path = "../datasets/Aminer-WhoIsWho (na-v3)/train_author.json"
pubs_output_path = "../datasets/Aminer-WhoIsWho (na-v3)/processed/pubs_enriched.json"
auths_output_path = "../datasets/Aminer-WhoIsWho (na-v3)/processed/authors_extracted.json"



# f = open(auths_path)
# keys = json.load(f)
# for key in keys:
#     n_auth = len(keys[key])
#     if n_auth > 4:
#         continue
#     found = True
#     for author in keys[key]:
#         n_papers = len(keys[key][author])
#         if n_papers > 100:
#             found = False
#     if found==True:
#         print(key)
#
# exit()


#read json file into a dictionary
f = open(pubs_path)
pubs = json.load(f)
pubs_list = []
for pub in pubs:
    pubs_list.append(pubs[pub])
pubs_rdd = sc.parallelize(pubs_list).map(lambda x: (x['id'], x))

f = open(auths_path)
keys = json.load(f)
author_info = []
for key in keys:
    for author in keys[key]:
        for paper in keys[key][author]:
            author_info.append(dict(id=paper, key=key, aid=author))
author_info_rdd = sc.parallelize(author_info).map(lambda x: (x['id'], dict(aid=x['aid'], key=x['key']))).groupByKey().map(lambda x: (x[0], list(x[1])))

publications_rdd = pubs_rdd.leftOuterJoin(author_info_rdd).map(add).map(lambda x: x[1][0]).map(injectID)
authors_rdd = publications_rdd.flatMap(pubToAuths)

publications_rdd.map(json.dumps).saveAsTextFile(pubs_output_path)
authors_rdd.map(json.dumps).saveAsTextFile(auths_output_path)

print("Number of authors: %s" % authors_rdd.count())
print("Number of publications: %s" % publications_rdd.count())
