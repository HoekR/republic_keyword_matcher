#from .finders.parse_delegates import *
from .models.models import TextWithMetadata
#from .settings import base_config
from republic.elastic.republic_elasticsearch import initialize_es
#es_config = set_elasticsearch_config(host_type='external')

es_republic = initialize_es(host_type='external')

# #### Participant_list
#es = Elasticsearch()

def get_presentielijsten(year: str='0', index: str='republic_pagexml_meeting', es=es_republic):
    prs_body = {
            "query": {
                "term":
                    {"year": year}
            },
            "size": 5000,
            "sort": ["_id"],

        }

    presentielijsten = {}
    presentielijsten_results = es.search(index=index, body=prs_body)

    for ob in presentielijsten_results['hits']['hits']:
        try:
            mt = TextWithMetadata(ob)
            presentielijsten[mt.id] = mt
        except AttributeError:
            print (ob)
    return presentielijsten




def get_nihil_actum(es=es_republic, index='republic_paragraphs'):
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


    na_results = es.search(index=index, body=na_body)
    nihil_actum = {}
    for ob in na_results["hits"]["hits"]:
        if ob['_source']['metadata']['paragraph_id']:
            mt = TextWithMetadata(ob)
            nihil_actum[mt.para_id] = mt
    return nihil_actum

# for n in nihil_actum.keys():
#     nad = nihil_actum[n]
#     print(nad.get_meeting_date(), ": ", n)
