i
import republic.parser.republic_file_parser as file_parser
from republic.config.republic_config import base_config, set_config_year


from elasticsearch import Elasticsearch
import republic.elastic.republic_elasticsearch as rep_es

es = Elasticsearch()


def query_delegates(es):

    body={
      "query": {
        "bool": {
          "must": [
            {
              "term": {
                "page_num": "323"
              }
            },
            {
              "match": {
                "page_side": "odd"
              }
            }
          ]
        }
      }
    }
    results = es.search(index="republic_hocr_pages", body=body)

results = query_delegates(es)
pr_results = [TextWithMetadata(ob) for ob in results["hits"]["hits"]]
pr_results[:5]
