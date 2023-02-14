import hashlib
import json
import os, glob
from utility import *
from pyspark import SparkConf
from pyspark import SparkContext
import string


def clean_string(s):
    return s.translate(str.maketrans('', '', string.punctuation)).lower()


def list_to_file(dict_list, file_path):
    with open(file_path, 'a') as file:
        file.truncate(0)
        for elem in dict_list:
            json.dump(elem, file, ensure_ascii=False)
            file.write("\n")


def clear_directory(dir):
    for file in os.scandir(dir):
        os.remove(file.path)


def file_to_list(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def lnfi(s):
    """
    Return the clustering key for the author name (Last Name - First Initial).
    Argument:
        s: the string indicating the author name
    Returns:
        key: the key
    """
    fullname = s.split(" ")
    if len(fullname) == 1:
        return str(''.join(ch for ch in "".join(fullname).lower() if ch.isalnum()))
    if len(fullname) == 2:
        try:
            return str(''.join(ch for ch in fullname[0][0].lower() if ch.isalnum())) + str(''.join(ch for ch in fullname[1].lower() if ch.isalnum()))
        except:
            return str(''.join(ch for ch in fullname[1].lower() if ch.isalnum()))
    if len(fullname) > 2:
        surname = [fullname[len(fullname) - 1]]
        name = fullname[0:len(fullname) - 1]
        res = " ".join(name).lower()[0] + " ".join(surname).lower()
        return str(''.join(ch for ch in res if ch.isalnum()))


def data_field_stats(data, field, field_type, output_file):
    """
    Count frequencies of terms in a given field.
    Arguments:
        data: the data to be processed
        field: the field to be considered in the computation of the statistics
        field_type: type of the field to be processed (string or list)
        output_file: location of the output file
    Returns:
        stats: the frequencies of each field value
    """
    if field_type == "string":
        stats = data.map(lambda x: clean_string(x[field])).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1]).collect()
    else:
        stats = data.flatMap(lambda x: list(map(lambda e: clean_string(e), x[field]))).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1]).collect()

    list_to_file(stats, output_file)
    return stats
