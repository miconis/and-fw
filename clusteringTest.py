# from sklearn import metrics
# labels_true = [0, 0, 0, 1, 1, 1]
# labels_pred = [0, 0, 1, 1, 2, 2]
# res = metrics.rand_score(labels_true, labels_pred)
# print(res)
# res = metrics.adjusted_rand_score(labels_true, labels_pred)
# print(res)
#
# labels_true = [0, 0, 0, 0, 0, 0]
# labels_pred = [1, 1, 1, 1, 1, 1]
# res = metrics.rand_score(labels_true, labels_pred)
# print(res)
# res = metrics.adjusted_rand_score(labels_true, labels_pred)
# print(res)
#
# labels_true = [1, 1, 1, 2, 2, 2]
# labels_pred = [0, 0, 0, 0, 0, 0]
# res = metrics.rand_score(labels_true, labels_pred)
# print(res)
# res = metrics.adjusted_rand_score(labels_true, labels_pred)
# print(res)

def fullnameFormatting(s):
    fullname = s.split(" ")
    if len(fullname) == 1:
        return fullname
    if len(fullname) == 2:
        return fullname[0] + ", " + fullname[1]
    if len(fullname) > 2:
        name = [fullname[len(fullname) - 1]]
        surname = fullname[0:len(fullname) - 1]
        return " ".join(surname) + ", " + " ".join(name)

def LNFI(fullname):
    fullname = fullnameFormatting(fullname)
    try:
        surname = ''.join(ch for ch in fullname.split(", ")[0].lower() if ch.isalnum())
        name = fullname.split(", ")[1].lower()
        return str(surname+name[0])
    except:
        return str(''.join(ch for ch in "".join(fullname).lower() if ch.isalnum()))

def clusterStats(c):
    #c: list of ids in the cluster
    ret = dict()
    ids = list(dict.fromkeys(c))
    for id in ids:
        ret[id] = c.count(id)
    return ret

def clustersFlatMapper(c):
    ret = []
    for key in c[1]:
        ret.append((key, c[1][key]))
    return ret

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
import json

conf = SparkConf().setAppName('AMiner-WhoIsWho Stats').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

authors = sc.textFile("../datasets/Aminer-WhoIsWho (na-v3)/processed/authors_extracted.json").map(json.loads)

#CLUSTERING PROCESS WITH LNFI
clustersLNFI = authors.map(lambda a: (LNFI(a['name']), [a['id']])).reduceByKey(lambda x, y: x + y).mapValues(lambda c: clusterStats(c)).flatMap(lambda c: clustersFlatMapper(c)).reduceByKey(lambda x, y: max(x, y))
#se alla fine faccio un reducebykey viene male, percheé somma tutti i cluster e quindi eè ovvio che il risultato finale sia che sono tutti uguali e zero sbagliati
#trovare un'unitaà di misura che non faccia semplicemente la somma

#CLUSTERING PROCESS GROUND TRUTH
clustersGT = authors.map(lambda a: (a['id'], 1)).reduceByKey(lambda x, y: x + y)

res = clustersGT.fullOuterJoin(clustersLNFI)

diff = res.filter(lambda x: x[1][0] != x[1][1]).count()
same = res.filter(lambda x: x[1][0] == x[1][1]).count()

print("different %s" % diff)
print("same %s" % same)

# SAME RESULT AS THE GROUND TRUTH, NO ELEMENT IS MISSING

# for i in res.take(10):
#     print(i)
