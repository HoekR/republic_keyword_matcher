#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import Counter

from ..fuzzy.fuzzy_keyword_searcher import score_levenshtein_distance_ratio
from .elastic_search_helpers import make_presentielijsten
from .finders.finders import *
from .finders.parse_delegates import *
from .helpers.vars2graph import vars2graph



fuzzysearch_config = {
    "char_match_threshold": 0.8,
    "ngram_threshold": 0.6,
    "levenshtein_threshold": 0.5,
    "ignorecase": False,
    "ngram_size": 2,
    "skip_size": 2,
}

with open('./data/json/republic_junk.json','r') as rj:
    ekwz = json.load(fp=rj)
provincies=['Holland','Zeeland','West-Vriesland','Gelderland','Overijssel', 'Utrecht','Friesland']
from republic.model.republic_phrase_model import month_names_early, month_names_late
months = month_names_early + month_names_late
junksweeper = FuzzyKeywordSearcher(config=fuzzysearch_config)
junksweeper.index_keywords(list(ekwz.keys()))
for key in ekwz.keys():
    for variant in ekwz[key]:
        junksweeper.index_spelling_variant(keyword=key, variant=variant)
junksweeper.index_keywords(months)
junksweeper.index_keywords(provincies)


def sweep_list(dralist, junksweeper=junksweeper):
    def get_lscore(r):
        return r['levenshtein_distance']

    rawres = []
    for t in dralist:
        #    t = rawlist[i]
        if len(t) > 1:
            rawtext = ' '.join(t)
        else:
            rawtext = t[0]
        r = junksweeper.find_candidates(rawtext)
        try:
            nr = max(r, key=get_lscore)
            if nr['levenshtein_distance'] < 0.5:
                rawres.append(t)
        except ValueError:
            rawres.append(t)
        return rawres

def list_tograph(list: list, fks: object):
    cl_heren = fks.find_close_distance_keywords(heren)
    G_heren = nx.Graph()
    d_nodes = sorted(cl_heren)
    for node in d_nodes:
        attached_nodes = cl_heren[node]
        G_heren.add_node(node)
        for nod in attached_nodes:
            G_heren.add_edge(node, nod)

abbreviated_delegates = pd.read_pickle('sheets/abbreviated_delegates.pickle')



def run(year, fuzzysearchconfig=fuzzysearch_config, junksweeper=junksweeper):
    make_presentielijsten(year=year)
    searchobs = make_presentielijsten(year=year)
    print ('year: ', len(searchobs), 'presentielijsten')
    print ("1. finding presidents")
    presidents = president_searcher(presentielijsten=searchobs)
    print(len(presidents), 'found')
    heren = [h.strip() for h in presidents]
    print("2.find provincial extraordinaris gedeputeerden")
    ps = province_searcher(presentielijsten=searchobs)
    print("finding unmarked text")
    unmarked = make_groslijst(presentielijsten=searchobs)
    c = Counter(unmarked)
    filtered_text = " ".join(list(c.keys()))
    fks = FuzzyKeywordSearcher(fuzzysearch_config)
    tussenkeys = fks.find_close_distance_keywords(list(c.keys()))
    dralist = vars2graph(tussenkeys)
    deputies = sweep_list(dralist)
# In[34]:




# In[35]:


#previously_matched = pd.read_excel('sheets/xml_herkend.xlsx')
previously_matched = pd.read_pickle('sheets/found_deputies.pickle')
previously_matched['id'] = previously_matched['ref_id']
previously_matched.columns

pm_heren = list(previously_matched['name'].unique())


# In[36]:


TUSSENVOEGSELS = [
    'à',
    '\xe0',
    r'd\'',
    r'de',
    r'den',
    r'der',
    r'des',
    r'di',
    r'en', 
    r'het',
    r"in 't",
    r'la',
    r'la', 
    r'le',
    r'of',
    r'van',
    r'ten', 
    r'thoe', 
    r'tot', 
    r"'t",
]


# In[37]:


stopwords = TUSSENVOEGSELS
def nm_to_delen(naam, stopwords=stopwords):
    nms = [n for n in naam.split(' ') if n not in stopwords]
    #nms.append(naam)
    return nms
           
keywords = list(abbreviated_delegates.name)
kwrds = {key:nm_to_delen(key) for key in keywords}


# In[38]:


nwkw = {d:k for k in list(set(keywords)) for d in k.split(' ') if d not in stopwords}


# In[39]:


# note that this config is more lenient than the default values 
fuzzysearch_config = {'char_match_threshold': 0.7,
 'ngram_threshold': 0.5,
 'levenshtein_threshold': 0.5,
 'ignorecase': False,
 'ngram_size': 2,
 'skip_size': 2}


# In[40]:


herensearcher = FuzzyKeywordSearcher(config=fuzzysearch_config)
exclude = ['Heeren', 'van', 'met', 'Holland']
filtered_kws = [kw for kw in nwkw.keys() if kw not in exclude]
herensearcher.index_keywords(keywords=filtered_kws)
for k in filtered_kws:
    for variant in nwkw[k]:
        herensearcher.index_spelling_variant(k,variant=variant)


# In[41]:


from collections import defaultdict
transposed_graph = defaultdict(list)
for node, neighbours in kwrds.items():
    for neighbour in neighbours:
        transposed_graph[neighbour].append(node)


# In[42]:


day = 1728
year = day


# In[43]:



dayinterval = pd.Interval(day, day, closed="both")
register = {}
pats = []


# In[44]:


# setup presidents for matching
existing_herensearcher = FuzzyKeywordSearcher(config=fuzzysearch_config)
existing_herensearcher.index_keywords(pm_heren)
connected_presidents = list(nx.connected_components(G_heren))
matchfnd = FndMatch(year=year,
                 #patterns=pats,
                 register=register,
                 rev_graph=transposed_graph,
                 searcher=herensearcher,
                 junksearcher=junksweeper,
                 df=abbreviated_delegates)

"""
- input is grouped delegates 
- output is matched delegates and unmatched 
"""
def find_delegates(input=[],
                  matchfnd=matchfnd,
                  df=abbreviated_delegates,
                  previously_matched=previously_matched,
                  year=1728):
    #matched_heren = defaultdict(list)
    matched_deputies = defaultdict(list)
    unmatched_deputies = []
    for herengroup in input:
        # we add the whole group to recognized if one name has a result
        recognized_group = []
        keyword_counter = Counter()
        in_matched = False
        for heer in herengroup: # we try to fuzzymatch the whole group and give the string a score
            rslt = matchfnd.match_candidates(heer=heer)
            if rslt:
                in_matched = True
                match_kw = getattr(rslt,'match_keyword')
                match_distance = getattr(rslt,'levenshtein_distance')
                recognized_group.append((heer, match_kw, match_distance))
                keyword_counter.update([match_kw])
            else:
                recognized_group.append((heer, '', 0.0))
        if in_matched == True: # if there is at least one match, proceed
            kw = keyword_counter.most_common()[0][0]
            rec = previously_matched.loc[previously_matched.name == kw]
            #ncol = 'proposed_delegate'
            ncol = 'name'
            if len(rec) == 0:
                rec = iterative_search(name=kw, year=year, df=df)
                ncol = 'name'
            drec = rec.to_dict(orient="records")[0]
            m_id = drec['id']
#             if m_id == 'matched':
#                 print(rec, kw)
            name = drec[ncol]
            score = drec.get('score') or 0.0
            matched_deputies[m_id] = {'id': m_id, 
                                      'm_kw':kw,
                                      'score':score,
                                      'name':name, 
                                      'variants':recognized_group}
        else:
            unmatched_deputies.append(herengroup) # non-matched deputy-groups are also returned 
    return({"matched":matched_deputies,
            "unmatched":unmatched_deputies})
# In[45]:


found_presidents = find_delegates(input=connected_presidents,
                                  matchfnd=matchfnd,
                                  df=abbreviated_delegates,
                                  previously_matched=previously_matched,
                                  year=1728)


# ### Cluster deputies

# We now do the same trick with the delegates. 

# In[47]:


found_delegates = find_delegates(input=deputies,
                                 matchfnd=matchfnd,
                                 df=abbreviated_delegates,
                                 previously_matched=previously_matched,
                                 year=1728 )


# ### join presidents and deputies
# 

# In[48]:


all_matched = {}
for d in [found_presidents['matched'], found_delegates['matched']]:
    for key in d:
        if key not in all_matched.keys():
            all_matched[key]= d[key]
        else:
            all_matched[key]['variants'].extend(d[key]['variants'])


# In[49]:


moregentlemen = [MatchHeer(all_matched[d]) for d in all_matched.keys() if type(d)==int] # strange keys sneak in


# In[50]:


mh_df = pd.DataFrame(all_matched).transpose().reset_index()


# In[51]:


matchsearch = FuzzyKeywordSearcher(config=fuzzysearch_config)
kws = defaultdict(list)
matcher = {}
for key in all_matched:
    variants = all_matched[key].get('variants')
    keyword = all_matched[key].get('m_kw')
    idnr = all_matched[key].get('id')
    name = all_matched[key].get('name')
    for variant in variants:
        kws[keyword].append(variant[0])
        matcher[variant[0]] = {'kw':keyword, 'id':idnr, 'name':name}
matchsearch.index_keywords(list(kws.keys()))
for k in kws:
    for v in kws[k]:
        matchsearch.index_spelling_variant(keyword=k,variant=v)


# In[52]:


deputyregister = defaultdict(list)
search_results = {}
# fm = FndMatch(year=1728,
#                  #patterns=pats,
#                  register=register,
#                  rev_graph=transposed_graph,
#                  searcher=herensearcher,
#                  junksearcher=junksweeper,
#                  df=abbreviated_delegates)
for T in searchobs:
    ob = searchobs[T].matched_text
    mo = MatchAndSpan(ob, junksweeper=junksweeper, previously_matched=previously_matched, match_search=matchsearch)
#     for span in ob.spans:
#         if span.type in ['president', 'delegate']:
#             if span.pattern == '':
#                 span.set_pattern(pattern=ob.item[span.begin:span.end])
#             r = fm.match_candidates(heer=span.pattern)
#             span.set_delegate(delegate_id=r.id, 
#                               delegate_name=r.proposed_delegate, 
#                               delegate_score=r.score)


# ### 8. Mark delegates 

# In[53]:


delresults = Counter()
for T in searchobs.keys():
    searchob = searchobs[T]
    result = get_delegates_from_spans(searchob.matched_text)
    try:
        id  = result['delegate']['id']
        if id:
            delresults.update([id])
    except KeyError:
        pass
delresults

previously_matched.loc[previously_matched.id.isin(delresults.keys())]
abbreviated_delegates.loc[abbreviated_delegates.id.isin(delresults.keys())]


# In[54]:


framed_gtlm = pd.DataFrame(moregentlemen)


# In[55]:


framed_gtlm.rename(columns={0:'gentleobject'}, inplace=True)

framed_gtlm.to_pickle('sheets/framed_gtlm.pickle')
# In[56]:


framed_gtlm['vs'] = framed_gtlm.gentleobject.apply(lambda x: [e for e in x.variants['general']])
framed_gtlm['ref_id'] = framed_gtlm.gentleobject.apply(lambda x: x.heerid)
#framed_gtlm['uuid'] = framed_gtlm.gentleobject.apply(lambda x: x.get_uuid())
framed_gtlm['name'] = framed_gtlm.gentleobject.apply(lambda x: x.name)
framed_gtlm['found_in'] = day
framed_gtlm


# In[57]:


def levenst_vals(x, txt):
    scores= [score_levenshtein_distance_ratio(i.form,txt) for i in x if len(i.form)>1]
    if max(scores) > 0.8:
        return max(scores)
    else:
        return np.nan


# In[58]:


def delegates2spans(searchob):
    spans = searchob.matched_text.spans
    for span in spans:
        txt = searchob.matched_text.item[span.begin:span.end]
        msk = framed_gtlm.vs.apply(lambda x: levenst_vals(x, txt))
        mres = framed_gtlm.loc[msk==msk.max()]
        if len(mres)>0:
    #     if len(mres)>0:
    #         setattr(span, 'delegate_id', mres.uuid.iat[0])
    #         setattr(span, 'delegate_name', mres.name.iat[0])
        #print(kand, all_matched[kand])
            span.set_pattern(txt)
        if len(mres)>0:
            span.set_delegate(delegate_id=mres.ref_id.iat[0], delegate_name=mres.name.iat[0])


# In[59]:


for T in searchobs:
    searchob = searchobs[T]
    #mo = MatchAndSpan(ob, junksweeper=junksweeper, previously_matched=previously_matched)
    delegates2spans(searchob)


# In[60]:


merged_deps = pd.merge(left=framed_gtlm, right=abbreviated_delegates, left_on="ref_id", right_on="id", how="left")


# In[61]:


serializable_df = merged_deps[['ref_id','geboortejaar', 'sterfjaar', 'colleges', 'functions',
       'period', 'sg', 'was_gedeputeerde', 'p_interval', 'h_life', 'vs','name_x', 'found_in',]]
serializable_df.rename(columns={'name_x':'name', 
                                'vs':'variants', 
                                "h_life": "hypothetical_life", 
                                "p_interval":"period_active", 
                                "sg": "active_stgen"},  inplace=True)


# In[62]:


serializable_df.to_pickle('sheets/found_deputies.pickle')


# In[63]:


for interval in ["hypothetical_life", "period_active"]:
    serializable_df[interval]=serializable_df[interval].apply(lambda x: [x.left, x.right])
serializable_df.to_excel('sheets/found_deputies.xlsx')


# In[64]:


out=[]
for T in searchobs:
    ob = searchobs[T]
    out.append(ob.to_dict())


# In[65]:


#with open('/Users/rikhoekstra/Downloads/1728_pres2.json', 'w') as fp:
json_year = json.dumps(obj=out)


# In[ ]:




jsondelegates.rename(columns={"h_life": "hypothetical_life", "p_interval":"period_active", "sg": "active_stgen"},
                     inplace=True)
# In[66]:


delegate_json = serializable_df[['ref_id', 'name', 'geboortejaar', 'sterfjaar', 'colleges',
       'functions', 'active_stgen', 'was_gedeputeerde',
       'period_active', 'hypothetical_life']].to_json(orient="records")

with open("/Users/rikhoekstra/Downloads/delegats.json", 'w') as fp:
    fp.write(delegate_json)
# In[67]:


from IPython.display import HTML
htmlout=[]
for T in searchobs:
    ob = searchobs[T].matched_text
    url = searchobs[T].make_url()
    ob.mapcolors()
    rest = ob.serialize()
    rest = f"\n<h4>{T}</h4>\n" + rest
    if url:
        rest += f"""<br/><br/><a href='{url}'>link naar {T}-image</a><br/>"""
    htmlout.append(rest)
#out.reverse()
HTML("<br><hr><br>".join(htmlout))

from IPython.display import HTML
htmlout=[]
for T in searchobs:
    ob = searchobs[T].matched_text
    url = searchobs[T].make_url()
    ob.mapcolors()
    rest = ob.serialize()
    rest = f"\n<h4>{T}</h4>\n" + rest
    if url:
        rest += f"""<br/><br/><a href='{url}'>link naar {T}-image</a><br/>"""
    htmlout.append(rest)
#out.reverse()
HTML("<br><hr><br>".join(htmlout))

# In[68]:


from IPython.display import HTML
t_out=[]
for T in searchobs:
    ob = searchobs[T].matched_text
    url = searchobs[T].make_url()
    ob.mapcolors()
    rest = ob.serialize()
    rest = f"""\n<tr><td><strong>{T}</strong></td><td>{rest}</td>"""
    if url:
        rest += f"""<td><a href='{url}'>link naar {T}-image</a></td>"""
    rest += "</tr>"
    t_out.append(rest)
#out.reverse()
outtable = "".join(t_out)
HTML(f"<table>{outtable}</table>")


# In[69]:


with open(f"/Users/rikhoekstra/Downloads/{day}_check.html", 'w') as flout:
    flout.write(f"<html><body><h1>results for {day}</h1>\n")
    flout.write(f"<table>{outtable}</table>")
    flout.write("</body></html>")

for T in searchobs:
    search_results = {}
    ob = searchobs[T].matched_text
    unmarked_text = ''.join(ob.get_unmatched_text())
    splittekst = re.split(pattern="\s",string=unmarked_text)
    for s in splittekst:
        if len(s)>2 and len(junksweeper.find_candidates(s))==0:
            sr = identified.get(s)
            try:
                if len(sr) > 0:
                    sr = sr.loc[sr.score == sr.score.max()]
                    nm = sr.name.iat[0]
                    idnr = sr.id.iat[0]
                    score = sr.score.max()
                    b = ob.item.find(s)
                    e = b + len(s)
                    span = ob.set_span(span=(b,e), clas='delegate', pattern=s, delegate=idnr, score=score)
                    search_results.update({s:{'match_term':nm, 'match_string':s, 'score': score}, 'spandid':span})
            except TypeError:
                pass# "naam": {
#                         "properties":{
#                             'geslachtsnaam': {"type":"string"},
#                             'fullname': {"type":"string"},
#                             'birth': {"type":"string"},
#                             'death': {"type":"string"},
#                             'by': {"type":"integer"},
#                             'dy': {"type":"integer"},
#                             'bp': {"type":"string"},
#                             'dp': {"type":"string"},
#                             'biography': {"type":"text"},
#                             'reference': {"type":"string"}
#                         }
#                     },
#         }

att_lst_mapping = {
    "mappings": {
        "properties": {
                 'metadata':{"properties":{
                           "coords":{"properties":{
                                 'bottom': {"type":"int"},
                                 'height': {"type":"int"},
                                 'left': {"type":"int"},
                                 'right': {"type":"int"},
                                 'top': {"type":"int"},
                                 'width': {"type":"int"}
                               },
                            'inventory_num':{"type": "int"},
                            'meeting_lines':{"type": "string"}, 
                            'text':{"type":"string"},
                            'zittingsdag_id':{"type":"string"}, 
                            'url':{"type":"string"}
                           }
                         }
                        }
                    },
                 'spans': {"properties":{
                           'offset':{"type":"integer"},
                             'end':{"type":"integer"},
                             'pattern':{"type":"string"},
                             'class':{"type":"string"},
                             'delegate_id':{"type":"string"},
                             'delegate_name':{"type":"string"},
                             'delegate_score':{"type":"float"}}
                           }
            }
    }

# ## To Elasticsearch

# In[70]:


from republic.republic_keyword_matcher.elastic_search_helpers import bulk_upload
local_republic_es = Elasticsearch()
local_republic_es


# In[ ]:


local_republic_es.indices.create(index='attendancelist')


# In[ ]:


data_dict = out
bulk_upload(bulkdata=data_dict, index='attendancelist', doctype='attendancelist')

