from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
import json

dataset_path = "/Users/miconis/Desktop/Authors Disambiguation/datasets/ni"

conf = SparkConf().setAppName('Neuroinformatics Stats').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

ni = sc.textFile(dataset_path).map(json.loads).filter(lambda x: "dedup" not in x['id'])
ni_size = ni.count()

ni_dataset = ni.filter(lambda x: 'dataset' in x['type'])
ni_dataset_size = ni_dataset.count()

ni_software = ni.filter(lambda x: 'software' in x['type'])
ni_software_size = ni_software.count()

ni_publication = ni.filter(lambda x: 'publication' in x['type'])
ni_publication_size = ni_publication.count()

ni_other = ni.filter(lambda x: 'other' in x['type'])
ni_other_size = ni_other.count()

print("Neuroinformatics size: " + str(ni_size))
print("Publication size: " + str(ni_publication_size))
print("Other size: " + str(ni_other_size))
print("Software size: " + str(ni_software_size))
print("Dataset size: " + str(ni_dataset_size))


print(ni_publication.take(1)[0])