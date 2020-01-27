import re

from settings import base_config
from collections import Counter
from utils import *
from republic.fuzzy import FuzzyKeywordSearcher


# #### praesentibus


def make_alternatives(term, searcher, alternatives=[], matchlist=presentielijsten, fromscratch=False):
    """

    :param searcher: FuzzyKeywordSearcher
    :type matchlist: dictionary
    :type alternatives: list
    """
    searcher = FuzzyKeywordSearcher(config=parse_delegates.base_config)
    alternatives = alternatives

    for T in matchlist.keys():
        res = searcher.find_candidates_new(keyword=term, text=matchlist[T].text)
        if res:
            alternatives.append(res[0]['match_string'])
    alts = list(set(alternatives))
    for variant in alts:
        searcher.index_spelling_variant(term, variant)
    return searcher, alts


def president_searcher(presentielijsten, searcher):
    """search and mark president in delegate attendance list
    this returns heren, but marks the presidents in the presentielijsten texts"""
    president_searcher = searcher
    heren = []
    text_zonder_president = []  # for troubleshooting
    pat = "%s.*(Den Heere )(.*)%s"  # de presidenten
    pats = []
    for T in list(presentielijsten.keys()):
        txt = presentielijsten[T].text
        president = president_searcher.find_candidates_new(keyword='PRASIDE', include_variants=True, text=txt)
        presentibus = presentibus_searcher.find_candidates_new(keyword='PRAESENTIBUS', include_variants=True, text=txt)
        begin = 0  # in case we find no president marker
        end = 0
        try:
            ofset = president[0]['match_offset'] or 0
            end = presentibus[0]['match_offset'] or len(txt)
            prez = president[0]['match_string'] or ''
            prae = presentibus[0]['match_string'] or ''
            searchpat = pat % (re.escape(prez), re.escape(prae))
            r = re.search(searchpat, txt)
            if r and r.group(2):
                heer = r.group(2).strip()
                heer = re.sub('[^\s\w\.]*', '', heer)
                heren.append(heer)
                mt = presentielijsten[T].matched_text
                span = get_span_from_regex(r)
                mt.set_span(span, "president")
                setattr(mt, 'found', {'president': heer})
            else:
                text_zonder_president.append(T)
                pats.append((searchpat, txt))

        except IndexError:
            text_zonder_president.append(T)

        return heren


def province_searcher( presentielijsten, config=base_config):
    kw_searcher = FuzzyKeywordSearcher(config)
    province_order = ["gelderlandt",
                      "hollandt ende west-frieslandt",
                      "utrecht",
                      "frieslandt",
                      "overijssel",
                      "groningen",
                      "zeelandt"
                     ]

    searchstring = "met {} extraordinaris gedeputeerde uyt de provincie van {}"
    provinces = []
    for provincie in province_order:
        for telwoord in ['een', 'twee', 'drie']:
            provinces.append(searchstring.format(telwoord, provincie))
    kw_searcher.index_keywords(provinces)

    for T in presentielijsten.keys():
        itm = presentielijsten[T]
        txt = itm.text
        mt = itm.matched_text
        for kw in provinces:
            provs = kw_searcher.find_candidates_new(keyword=kw, text=txt)
        if len(provs) > 0:
            profset = provs[0].get('match_offset') or 0
        for res in provs:
            ofset = res['match_offset']
            span = (ofset, ofset + len(res['match_string']))
            mt.set_span(span, "province")


def make_groslijst(presentielijsten):
    """get rough list of unmarked text from presentielijsten"""
    groslijst_ = []
    interpunctie = re.compile("[;:,\. ]+")
    unmarked_texts = {}
    for T in presentielijsten:
        marked_item = presentielijsten[T].matched_text
        unmatched_texts = marked_item.get_unmatched_text()
        unmatched_texts = [u.strip() for u in unmatched_texts if len(u.strip())>3]
        for unmatched_text in unmatched_texts:
            if len(unmatched_text)>3:
                interm = interpunctie.split(unmatched_text)
                interm = [i.strip() for i in interm if len(i.strip())>3]
                groslijst_.extend(interm)
    return groslijst

def find_unmarked_deputies(keywords):
    """mark deputies from keyword list """
    deputy_heuristic = FuzzyKeywordSearcher(config=base_config)
    deputy_heuristic.index_keywords(kws)
    deputy_heuristic.index_spelling_variants = True
    ll = {}
    deps = deputy_heuristic.find_close_distance_keywords(keyword_list=[w[0] for w in keywords])

    merge(deps.values())
    deputies = {d: deps[d] for d in kws}
    deputy_searcher = FuzzyKeywordSearcher(config=base_config)
    deputy_searcher.index_keywords(list(deputies.keys()))
    deputy_searcher.index_spelling_variants = True
    for k in deputies.keys():
        for val in deputies[k]:
            deputy_searcher.index_spelling_variant(k, val)
    for T in presentielijsten.keys():
        itm = presentielijsten[T]
        txt = itm.text
        mt = itm.matched_text
        marker(text=txt, item=mt, keywords=kws, searcher=deputy_searcher,
               itemtype="delegate2")