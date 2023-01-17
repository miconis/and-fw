import hashlib
import json

def test(x):
    print(x)

def list_to_file(dict_list, file_path):
    with open(file_path, 'a') as file:
        file.truncate(0)
        for elem in dict_list:
            json.dump(elem, file)
