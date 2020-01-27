from elasticsearch import Elasticsearch
from settings import base_config
from parse_delegates import *

# #### Participant_list
es = Elasticsearch()

def get_presentielijsten(config=base_config):
    prs_body = {
        "query": {
            "term": {
                "metadata.categories": "participant_list"
            }
        },
        "size": 5000,
        "sort": ["metadata.meeting_date"],

    }


    presentielijsten = {}
    presentielijsten_results = es.search(index="paragraph_index", body=prs_body)
    for ob in presentielijsten_results["hits"]["hits"]:
        mt = TextWithMetadata(ob)
        presentielijsten[mt.para_id] = mt
    return presentielijsten


def get_nihil_actum():
    na_body = {"query": {
        "bool": {
            "must": [
                {
                    "match": {
                        "metadata.keyword_matches.match_category": "non_meeting_date"
                        }
                    }
                ],

            }
        },
        "size": 5000,
        "sort": ["metadata.meeting_date"],
    }


    na_results = es.search(index="paragraph_index", body=na_body)
    nihil_actum = {}
    for ob in na_results["hits"]["hits"]:
        if ob['_source']['metadata']['paragraph_id']:
            mt = TextWithMetadata(ob)
            nihil_actum[mt.para_id] = mt
    return nihil_actum

# for n in nihil_actum.keys():
#     nad = nihil_actum[n]
#     print(nad.get_meeting_date(), ": ", n)
