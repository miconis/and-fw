# functions to process datasets of each provider: creates publications json files with author ground truth ids
from utility import *
import json


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


aminer_wiw_to_json(pubs_input_path="datasets/Aminer-WhoIsWho (na-v3)/train_pub.json",
                   authors_gt_path="datasets/Aminer-WhoIsWho (na-v3)/train_author.json",
                   pubs_output_path="datasets/processed/aminer_wiw_pubs.json")
