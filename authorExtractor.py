#script to extract authors from publications

import hashlib
import json

pubs_path = "../datasets/SCAD-zbMATH/scad-zbmath-01-open-access.json"
output_path = "../datasets/SCAD-zbMATH/authors-scad-zbmath-1.json"

#extract authors from publications

def pub2Authors(pub):
    authors = []
    for author in pub["authors"]:
        coauthors = pub["authors"]
        coauthors.remove(author)
        a = dict(
            id= hashlib.md5((str(author['id'])+str(pub['year'])+str(pub['title'])+str(pub['venue'])).encode("utf8")).hexdigest(),
            name= author['name'],
            shortname= author['shortname'],
            pid=author['id'],
            coauthors=coauthors,
            publication=dict(
                year=pub['year'],
                title=pub['title'],
                venue=pub['venue']
            )
        )
        authors.append(a)
    return authors

json_authors = []
with open(pubs_path, "r") as pubs_file:
    for pub in pubs_file:
        pub = json.loads(pub)
        json_authors.extend(pub2Authors(pub))

#save list to JSON file
with open(output_path, "w", encoding="utf8") as json_file:
    for x in json_authors:
        json.dump(x, json_file, ensure_ascii=False)
        json_file.write("\n")
