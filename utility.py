import hashlib
import json
import os, glob


def list_to_file(dict_list, file_path):
    with open(file_path, 'a') as file:
        file.truncate(0)
        for elem in dict_list:
            json.dump(elem, file)
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
