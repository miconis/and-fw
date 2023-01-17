#script to process and convert the XML dump (SCAD-zbMATH) to JSON

import xml.etree.ElementTree as ET
import json

dataset_path = "../datasets/SCAD-zbMATH/scad-zbmath-01-open-access.xml"
output_path = "../datasets/SCAD-zbMATH/scad-zbmath-01-open-access.json"

#process XML dataset
tree = ET.parse(dataset_path)
root = tree.getroot()

json_pubs = []
#iterate over publications
for p in root:

    authors = []
    for a in p.find("authors"):
        authors.append(a.attrib)

    json_pubs.append(dict(
        id=p.attrib['id'],
        title=p.find("title").text,
        venue=p.find("venue").text,
        year=p.find("year").text,
        authors=authors
    ))

#save list to JSON file
with open(output_path, "w", encoding="utf8") as json_file:
    for x in json_pubs:
        json.dump(x, json_file, ensure_ascii=False)
        json_file.write("\n")
