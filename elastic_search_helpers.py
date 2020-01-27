i
import republic.parser.republic_file_parser as file_parser
from republic.config.republic_config import base_config, set_config_year


from elasticsearch import Elasticsearch
import republic.elastic.republic_elasticsearch as rep_es

es = Elasticsearch()

def simple_search(es, input):
    body = {
      "query": {
        "bool": {
          "must": [
            {"fuzzy": {
              "text": {
                "value": input
              }
            }
            }
          ],
        }
      },
      "from": 0,
      "size": 1000,
      "sort": "_score"
    }
  
    results: list = es.search(index="paragraph_index", body=body)
    return results


def search_resolutions_query(text):
  "search the resolutions, not the presentielijsten (as far as they are marked as such"
    body = {
      "query": {
        "bool": {
          "must":
            [{
              "fuzzy":
                {"text":
                   {"value": text
                    }
                 }
            }
            ],
          "must_not": [
            {
              "term": {
                "metadata.categories": "participant_list"
              }
            }
          ],
        }
      },
      "from": 0,
      "size": 10000,
      "sort": [],
    }

    return body


def get_presentielijsten(es):
    "this is now for the whole index, adjust for a year"
    body = {
      "query": {
        "term": {
          "metadata.categories": "participant_list"
        }
      },
      "size": 5000,
      "sort": ["metadata.meeting_date"],

    }

    results: list = es.search(index="paragraph_index", body=body)

    return results
    


####################
# query namenindex #
####################

def get_name_from_namenindex(proposed):
  """get a name candidate from the namenindex.
  It does not keep count of a time frame as we only want a vote on the right name from the namenindex
  and assume that names in the namenindex tend to be uniform.
  """
  body = {
    "query": {
      "bool": {
        "must": [
          {"fuzzy": {
            "fullname": {
              "value": proposed.lower()
            }
          }
          }
        ],
      }
    },
    "from": 0,
    "size": 1000,
    "sort": "_score"
  }

  results = es.search(index="namenindex", body=body)
  candidate = None
  if results['hits']['total'] > 0:
    names = [r['_source']['geslachtsnaam'] for r in results['hits']['hits']]

    candidate = Counter(names).most_common(3)

  return candidate


def get_name_for_disambiguation(proposed, base_config, debug=False):
  """get a name candidate from the namenindex.
  It keeps count of a time frame and tries to select the most likely candidate.
  TODO make this more sophisticated
  and the time stuff syntax is apparently not good
  """
  low_date_limit = base_config['year'] - 85  # or do we have delegates that are over 85?
  high_date_limit = base_config['year'] - 20  # 20 is a very safe upper limit

  body = {
    "query": {
      "bool": {
        "must": [
          {
            "fuzzy": {
              "fullname": {
                "value": proposed.lower()
              }
            }
          },
          {
            "range": {
              "by": {"gt": low_date_limit,
                     "lt": high_date_limit
                     },
            }
          },
          {
            "range": {
              "dy": {"lt": base_config['year'] + 50,
                     "gt": base_config['year']
                     },
            }

          }
        ],
        "should": [
          {"match":
             {"collection":
                {"query": "raa",
                 "boost": 2.0
                 }
              }
           }
        ]
      }
    },
    "from": 0,
    "size": 1000,
    "sort": "_score"
  }

  results = es.search(index="namenindex", body=body)
  candidate = None
  if results['hits']['total'] > 0:
    names = [r['_source']['geslachtsnaam'] for r in results['hits']['hits']]
    people = [r['_source']['fullname'] for r in results['hits']['hits']]
    interjections = people = [r['_source']['intrapositie'] for r in results['hits']['hits']]
    candidate = [r for r in results['hits']['hits']]
  # think about this some more
  print('nr of candidates: {}'.format(results['hits']['total']))
  if debug == True:
    print(body)
  return candidate  # ([names, people])
