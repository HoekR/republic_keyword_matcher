#!/usr/bin/env python
import os
import sys
import re

from collections import defaultdict
import pandas as pd
import numpy as np

# brrrr
repo_dir = os.path.split(os.getcwd())[0]

if repo_dir not in sys.path:
    sys.path.append(repo_dir)
import networkx as nx
from statistics import mean
from collections import Counter
from republic.republic_keyword_matcher.finders import finders
from republic.republic_keyword_matcher import delegates_settings
from ..republic_keyword_matcher.helpers import visualize_as_html
from republic.republic_keyword_matcher.models import *
from republic.republic_keyword_matcher.helpers import vars2graph
from republic.republic_keyword_matcher.finders.parse_delegates import *
from republic.fuzzy.fuzzy_patterns import dutch_date_patterns
from republic.fuzzy.fuzzy_keyword_searcher import score_levenshtein_distance_ratio, score_char_overlap_ratio, score_ngram_overlap_ratio
from republic.republic_keyword_matcher import settings
from republic.elastic import republic_elasticsearch
es_republic = republic_elasticsearch.initialize_es(host_type="external", timeout=60)

fuzzysearch_config = {
    "char_match_threshold": 0.8,
    "ngram_threshold": 0.6,
    "levenshtein_threshold": 0.5,
    "ignorecase": False,
    "ngram_size": 2,
    "skip_size": 2,
}

presentielijsten = get_presentielijsten(es=es_republic, index='pagexml_meeting', year=1728)
searchobs = results_to_obs(presentielijsten)
presidents = president_searcher(presentielijsten=searchobs)
heren = presidents
heren = [h.strip() for h in heren]
ps = finders.province_searcher(presentielijsten=searchobs)
unmarked = finders.make_groslijst(presentielijsten=searchobs)
c = Counter(unmarked)
filtered_text = " ".join(list(c.keys()))
fks = FuzzyKeywordSearcher(fuzzysearch_config)
tussenkeys = fks.find_close_distance_keywords(list(c.keys()))
distractors = tussenkeys.keys()
bare_junk = pd.DataFrame().from_dict(c, orient="index").reset_index()
bare_junk.rename(columns={'index':'term'}, inplace=True)
bare_junkpresents = bare_junk.loc[bare_junk.term.apply(lambda x:score_levenshtein_distance_ratio('PRASENTIBUS', x) > 0.5)]
presents_u = presents.index
presents_ubare_junk.drop(presents_u, axis=0, inplace=True)
bare_junk
# ### 4.strip out junk as far as we know it .
# 
# And make the words in the lists unique
# Junk is kept in a separate file. This needs some thought

# Known junk definitions 
# 
# __TODO__: cleanup

# In[44]:


with open('republic_junk.json','r') as rj:
    ekwz = json.load(fp=rj)
provincies=['Holland','Zeeland','West-Vriesland','Gelderland','Overijssel', 'Utrecht','Friesland']
from republic.model.republic_phrase_model import week_day_names, month_names_early, month_names_late, spelling_variants
months = month_names_early + month_names_late 
junksweeper = FuzzyKeywordSearcher(config=fuzzysearch_config)
junksweeper.index_keywords(list(ekwz.keys()))
for key in ekwz.keys():
    for variant in ekwz[key]:
        junksweeper.index_spelling_variant(keyword=key, variant=variant)
junksweeper.index_keywords(months)
junksweeper.index_keywords(provincies)


# Filter out the words from the unmarked list where 
fks = FuzzyKeywordSearcher(fuzzysearch_config)
rawlst = fks.find_close_distance_keywords(filtered_text)
rawlist = {k:list(set(rawlst[k])) for k in rawlst.keys()}
junk_finder=FuzzyKeywordSearcher(fuzzysearch_config)
junk_finder.index_keywords(real_junk)
# In[45]:


# note these are grouped categories, not individual keywords
dralist = vars2graph(tussenkeys)


# In[46]:


def get_lscore(r):
    return r['levenshtein_distance']

rawres = []
for t in dralist:
#    t = rawlist[i]
    if len(t)>1:
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


# In[47]:


#deputies = fks.find_close_distance_keywords(rawlist)
deputies = rawres

rawres
# We now have:
# 
# 1. __heren__: a list of presidents
# 1. __ezkw__: a rough list of words that have been identified as junk before
# 1. __deputies__: a rough list of words that are presumably delegates

# ### 5. cluster them into connected variants. Ideally these would contain the different variants of deputies

# In[49]:


cl_heren = fks.find_close_distance_keywords(heren)
G_heren = nx.Graph()
d_nodes = sorted(cl_heren)
for node in d_nodes:
    attached_nodes = cl_heren[node]
    G_heren.add_node(node)
    for nod in attached_nodes:
        G_heren.add_edge(node, nod)
list(nx.connected_components(G_heren))

#which clusters them nicely


# In[50]:


import pandas as pd
from sqlalchemy import text


# In[51]:


abbreviated_delegates = pd.read_pickle('sheets/abbreviated_delegates.pickle')


# In[52]:


abbreviated_delegates["h_life"] = abbreviated_delegates.apply(lambda row: hypothetical_life(row), axis=1)


# In[53]:


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


# In[54]:


stopwords = TUSSENVOEGSELS
def nm_to_delen(naam, stopwords=stopwords):
    nms = [n for n in naam.split(' ') if n not in stopwords]
    #nms.append(naam)
    return nms
           
keywords = list(abbreviated_delegates.name)
kwrds = {key:nm_to_delen(key) for key in keywords}


# many of the found terms do not contain their prepositions, so we strip those from the names for matching.Furthermore names tend to be cut up in their parts, so we make a matcher that uses them and try to match with those. (with a reverse mapping to keep track of them)

# ### 6. Identify the attendance lists with this overview of deputies 

# In[55]:


nwkw = {d:k for k in list(set(keywords)) for d in k.split(' ') if d not in stopwords}


# In[56]:


# note that this config is more lenient than the default values 
fuzzysearch_config = {'char_match_threshold': 0.7,
 'ngram_threshold': 0.5,
 'levenshtein_threshold': 0.5,
 'ignorecase': False,
 'ngram_size': 2,
 'skip_size': 2}


# In[57]:


herensearcher = FuzzyKeywordSearcher(config=fuzzysearch_config)
exclude = ['Heeren', 'van', 'met', 'Holland']
filtered_kws = [kw for kw in nwkw.keys() if kw not in exclude]
herensearcher.index_keywords(keywords=filtered_kws)
for k in filtered_kws:
    for variant in nwkw[k]:
        herensearcher.index_spelling_variant(k,variant=variant)


# In[58]:


previously_matched = pd.read_excel('sheets/xml_herkend.xlsx')
previously_matched.rename(columns={'index':'name','p_id':'id'}, inplace=True)
previously_matched.columns


# In[59]:


pm_heren = list(previously_matched['name'].unique())


# In[60]:


list(nx.connected_components(G_heren))


# #### match by group: do we know this gentleman already?

# In[61]:


from collections import defaultdict
transposed_graph = defaultdict(list)
for node, neighbours in kwrds.items():
    for neighbour in neighbours:
        transposed_graph[neighbour].append(node)


# In[62]:


day = 1728
year = day


# In[63]:



dayinterval = pd.Interval(day, day, closed="both")
register = {}
pats = []


# In[64]:


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


# In[65]:


"""
- input is grouped delegates 
- output is matched delegates and unmatched 
"""
def find_delegates(input=[],
                  matchfnd=matchfnd,
                  df=abbreviated_delegates,
                  previously_matched=previously_matched,
                  year=1728):
    matched_heren = defaultdict(list)
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
            ncol = 'proposed_delegate'
            if len(rec) == 0:
                rec = iterative_search(name=kw, year=year, df=df)
                ncol = 'name'
            drec = rec.to_dict(orient="records")[0]
            m_id = drec['id']
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


# In[66]:


found_presidents = find_delegates(input=connected_presidents,
                                  matchfnd=matchfnd,
                                  df=abbreviated_delegates,
                                  previously_matched=previously_matched,
                                  year=1728)


# In[67]:


matched_presidents = found_presidents['matched']
matched_presidents


# In[ ]:




mh_df = pd.DataFrame(matched_presidents).transpose().reset_index()
mh_df
# ### Cluster deputies

# We now do the same trick with the delegates. 
deputies
# In[68]:


found_delegates = find_delegates(input=deputies,
                                 matchfnd=matchfnd,
                                 df=abbreviated_delegates,
                                 previously_matched=previously_matched,
                                 year=1728 )
matched_deputies = found_delegates['matched']

matched_deputiesgentlemen = [MatchHeer(matched_deputies[d]) for d in matched_deputies]framed_gentlemen = pd.DataFrame(gentlemen)
framed_gentlemen.rename(columns={0:'matchheer'}, inplace=True)deps_df = pd.DataFrame(matched_deputies).transpose().reset_index()
# ### join presidents and deputies
# 

# In[69]:


#overlap
[key for key in matched_deputies.keys() if key in matched_presidents.keys()]


# In[70]:


all_matched = {}
for d in [matched_presidents, matched_deputies]:
    for key in d:
        if key not in all_matched.keys():
            all_matched[key]= d[key]
        else:
            all_matched[key]['variants'].extend(d[key]['variants'])


# In[71]:


all_matched


# now we have all recognized names of deputies and presidents, but they sometimes overlap. The list of variants is very large in some cases and they do not all are homogeneous, so now we try to separate them into different groups.

# In[72]:


moregentlemen = [MatchHeer(all_matched[d]) for d in all_matched.keys()]


# ### 7. Discriminating delegates and variants
# 
# We now have collected presidents and delegates and joined them. Before processing further there is two more things to do:
# 
# - what delegates have we missed. We know that there are problems (1) in recognizing short names (example 'van Dam' epecially when recognizing without preposition) (2) properly recognizing names that also occur as parts of other names (example 'Hoorn' and 'Van Hoornebeek') and (3) possibly recognizing parts of names that may occur when names are hyphened (example 'Schim' 'melpennick'. hyphens are not recognized correctly). But it is hard to troubleshoot these now, so I think it is better to tackle these after applying the names to the attendants lists and see what goes wrong there.
# 
# - we have put recall before precision in collecting names. This has led to a variety of variants for delegate names. Some have only one variant,others many. Variants may be right but also obviously wrong (example: '-ex' for Vegelin). Another case is that some delegates have many variants but different 'proposed delegates' i.e. recognized name patterns. The Fuzzy Searcher can distinguish these names well. And we may also compare the proposed delegates between the MatchedHeer  objects. The goal is to get a reordered set of names, and prune obviously wrong patterns.
# 
# **Steps:**
# 
# 1. make a list of associated gentlemen and delegates
# 2. reorder the variants to most likely associations (depending on their (levenshtein) likeliness)
# 3. throw out non-variants, based on comparative length, likeliness with established distractor words (months mostly)
# 4. apply the definive list of gentlemen and variants to the attendance lists 
# 5. spot the non-recognized delegates
# 
# 
# 

# #### step make list of associated gentlemen and delegates

# In[73]:


associated = defaultdict(list)
for item in all_matched:
    pds=set([v[1] for v in all_matched[item]['variants']])
    for p in pds:
        associated[p].append(item)


# #### reorder the variants to most likely associates

# In[74]:


as_var = defaultdict(list)
for assoc in associated:
    da = associated[assoc]
    if len(da)>1:
        if assoc != '': # special case, deal with that later
            for hid in da:
                varis = [v[0] for v in all_matched[hid]['variants'] if v[1]==assoc]
                as_var[assoc].extend(varis)
        


# and see how they compare to separate clustered names o

# In[75]:


mh_df = pd.DataFrame(all_matched).transpose().reset_index()


# In[76]:


mh_df


# ### 8. Mark delegates 
ob = searchobs['meeting-1728-04-29-session-1']
t = ob.matched_text
t.get_unmatched_text()
t.item
# ### Number of sessions with and without presidents
# how many  zittingsdagen do we have with and without a president?

# In[77]:


met_pr = {}
zonder_pr = {}
for T in searchobs:
    ob = searchobs[T].matched_text
    pr = [s for s in ob.spans if s.type == 'president']
    if len(pr)==0:
        zonder_pr[T]=pr
    else:
        met_pr[T]=pr


# In[78]:


len(met_pr)


# In[79]:


len(zonder_pr)


# # TODO review the order of matching delegates

# The match and span object takes the unmarked text from the objects and tries to match them with a delegate using the code for identifying delegates. But the failures in the ocr complicates this so that we have to use isolated strings, instead of splitting the text by commas or periods and newlines. This  drops all sorts of prefixes and in the case of a composite name such as _Taats van Amerongen_ it also either makes a name match twice or match part of it with another name. Also the prefixes like _van_ can get matched to the wrong names. So what we will do is match the names and then return to the list of spans and see if we cannot make a better match with the whole names. We would also like  to include previously acquired knowledge for better matching, but we have to think about that more.
# Another thing is that we may overfit existing names, but we'll see about that.

# In[80]:


config = {'char_match_threshold': 0.7,
 'ngram_threshold': 0.6,
 'levenshtein_threshold': 0.6,
 'ignorecase': False,
 'ngram_size': 2,
 'skip_size': 2}

sps = FuzzyKeywordSearcher(config=config)
spns = [(s.pattern, s.get_delegate()) for s in ob.spans if s]
for s in spns:
    nm = s[1]['name']
    d_id = s
#    if nm == 'Taets van Amerongen':
    if sps.keyword_index.get(nm):
        sps.index_keywords(nm)


# In[81]:


spns

cand = sps.find_candidates(ob.matched_text.item)
cand = cand[0]
span = (cand["match_offset"], cand["match_offset"]+len(cand['match_string']))
ob.matched_text.set_span(clas='delegate', delegate_id=18838, delegate_name=cand['match_term'], span=span)
ob.matched_text.get_fragments()
# In[82]:


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


# In[83]:


deputyregister = defaultdict(list)
search_results = {}
fm = FndMatch(year=1728,
                 #patterns=pats,
                 register=register,
                 rev_graph=transposed_graph,
                 searcher=herensearcher,
                 junksearcher=junksweeper,
                 df=abbreviated_delegates)
for T in searchobs:
    ob = searchobs[T].matched_text
    mo = MatchAndSpan(ob, junksweeper=junksweeper, previously_matched=previously_matched, match_search=matchsearch)
    for span in ob.spans:
        if span.type in ['president', 'delegate']:
            if span.pattern == '':
                span.set_pattern(pattern=ob.item[span.begin:span.end])
            r = fm.match_candidates(heer=span.pattern)
            span.set_delegate(delegate_id=r.id, 
                              delegate_name=r.proposed_delegate, 
                              delegate_score=r.score)

for keyword in herensearcher.keyword_index:
    junk_is = junksweeper.find_candidates(text=keyword, include_variants=True)
    if junk_is:
        score = max(junk_is, key=lambda x: x['levenshtein_distance'])
        if score['match_keyword']==keyword:
            print(keyword, score)zs = [set(item.form) for item in vrts]
c = {s[0]:{'default':s[1],'variants':defaultdict(int)} for s in enumerate(n)}
for s in enumerate(zip(n, vrts[0].form)):
    compared = s[1]
    variants = c[s[0]]['variants']
    if compared[0]!=compared[1]:
        variants[compared[1]]+=1moregentlemen[7].variantsresults = {}
for heer in moregentlemen:
    vrts = heer.variants['general']
    results[heer.name] = [nw(heer.name,variant.form, mismatch=2, gap=0) for variant in vrts]resultslc = (0,'')
for item in results['van Isselmuden tot Zwollingerkamp']:
    i = item[1]
    l = len([c for c in i if c not in['-','.']])
    if l > lc[0]:
        lc = (l, i)
lccimport numpy as np

def nw(x, y, match = 1, mismatch = 1, gap = 1):
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx, nx + 1)
    F[0,:] = np.linspace(0, -ny, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            j -= 1
#     if ry[-1] != rx[-1]:
#         ry[-1] =  '-'
    # Reverse the strings.
    rx = ''.join(rx)[::-1]
    ry = ''.join(ry)[::-1]

    return [rx, ry]
    #return '\n'.join([rx, ry])from itertools import starmap
n = 'Van Singendonk'
setn = set(n)
sets = [set(item.form) for item in vrts]
list(starmap(setn.intersection, sets))
vrts
# In[93]:


matchfnd


# ## Continue matching

# In[115]:


defo = defaultdict(list)
for item in ob.get_unmatched_text():
    item = [i.strip() for i in re.split('[\.|,]', item)]
    for i in item:
        identified = ''
        res = {}
        r2 = {}
        for g in moregentlemen:
            mn = 0
            co = 0
            mc = 0
            vrts = g.variants['general']
            dr = [score_levenshtein_distance_ratio(x.form, i) for x in vrts if x.heerid==g.heerid]
            if len(dr)>0:
                mn = mean(dr)
            res[g] = {'mn':mn}
            if len(i)<5:
                co = [score_ngram_overlap_ratio(x.form, i, 1) for x in vrts if x.heerid==g.heerid]
                if co and len(co)>0:
                  mc = mean(co)
            res[g]['mc'] = mc
        best_l = max(res, key=lambda key: res[key]['mn'])
        best_c = max(res, key=lambda key: res[key]['mc'])
        if res[best_l]['mn'] > 0.5:
            identified = res[best_l]['mn']
            if best_c != 0:
                if res[best_c] != identified:
                    identified = res[best_c]
        if identified != '':
            delegate = fm.match_candidates(heer=best_l.name)
            identified = delegate
        else: 
            identified = None
        if identified:
            
            name = delegate.proposed_delegate
            m_id = delegate.id
        else:
            name = ''
            m_id = 0
        start = ob.item.find(i)
        end = start + len(i)
        ob.set_span(span=(start, end),pattern=i, clas='deputy', delegate_name=name,
                    delegate_id=m_id)
#         defo[i]={'mn':(best_l,res[best_l]['mn']),
#                  'co':(best_c,res[best_c]['mc'])}
# defo


# In[117]:


def get_mn(k, defo=defo):
    defo = dict(defo)
    v = defo[k]['mn'][1]
    v = float(v)
    return v
    
#         mn = dict(defo[k].get('mn').value
for k in defo:
    v = get_mn(k)
    if v < 0.5:
        res = iterative_search(name=k, year=1728, df=abbreviated_delegates)
        res


# In[119]:


s = moregentlemen[8]
print(s.variants)
for v in s.variants:
    print(score_levenshtein_distance_ratio('an Schwartzenberg', s.name))
    #list(map(lambda x: score_levenshtein_distance_ratio('an Schwartzenberg', x), s.variants))


# In[ ]:


v = s.variants['general'][0]
v.form


# In[ ]:


res

fr = ob.matched_text.get_fragments()

if prezidents:
    prez = prezidents[0]
    begin = prez['begin']
else:
    begin = 0
for fragment in fr:
    if fragment['type'] == 'unmarked' and fragment['end'] > begin:
        f = fragment['pattern']
        f == ''
        f = ob.matched_text.item[fragment['begin']:fragment[end]]
        fragments.append(f)
fragments
#ob.matched_text.get_unmatched_text()for span in ob.spans:
        if span.type in ['president', 'delegate']:
            if span.pattern == '':
                span.set_pattern(pattern=ob.item[span.begin:span.end])
            r = fm.match_candidates(heer=span.pattern)
            span.set_delegate(delegate_id=r.id, 
                              delegate_name=r.proposed_delegate, 
                              delegate_score=r.score)
print(ob.get_fragments())
# In[120]:


span.pattern


# In[121]:


# span.set_delegate(delegate_id=10, 
#                               delegate_name='blop', 
#                               delegate_score=1.0)
span.delegate_id


# In[122]:


all_matched.keys()


# In[123]:


previously_matched


# ### Some Evaluations (fwiw)

# In[124]:


pd.DataFrame(previously_matched.aggregate("mean")[['levenshtein_distance', 'score']])


# In[125]:


previously_matched.score.value_counts()


# In[126]:


previously_matched.levenshtein_distance.round(2).value_counts().sort_values('index', ascending=False)


# In[127]:


for s in ob.spans:
    print(s, s.idnr, ob.item[s.begin:s.end], s.get_delegate())


# ## a better searcher and identifier.
# 
# The problem: 
# We have two ways of identifying name patterns now.
# 
# - the first uses pandas based method to identify strings against a database of known delegates and then pulls in some of the external context. This works pretty well for initial identification, but not all names are recognized very well. There is especially a problem by names that have been badly garbled by the text recognition, which is now the case with the OCR, but it will probably not get better if we start using HTR and look up entities in the text. Another issue is that this method has no memory of earlier work, both in the sense that it will not know if earlier matches were found or earlier proposals were marked as false. 
# - the fuzzysearcher is very good for spotting name patterns in text as it includes approximate text matching and a number of variants. But it knows nothing about identification and there is no easy way of associating patterns with entities.
# - fuzzysearcher is also able to cluster alike word patterns and recluster heterogeneous lists. However, this does not per se bring us closer to identification
# - the only way to include previously selected variants is in the fuzzy searcher. Which is useful, but the main problem remains associating this with external knowledge.
# 
# This leads to the following requirements:
# - a way of associating patterns with identities. But how. Earlier I put the result of a Fuzzysearch pattern clustering in a dataframe, but this included heterogeneous patterns. One possibility is to revise candidates and recategorize weak candidates. The goals is to incrementally improve the patterns associated with a name. This is especially useful if we reuse the objects over many years.
# - another idea is to check for internal consistency of the results and try and find inconsistent results. This is especially useful for the presidents, as their number is even more limited than the delegates. The problem is overfitting in which unidentified names get associated with identified names and also new or infrequent delegates may drop out. SO there is a need for manual check.
# - another tool may be to make a specialized confusion matrix for delegates
# 
# We start with this latest
# no we don't. Most of this is already in place. The main problem is that we cannot access the it from 'both' sides, so that we have to rematch all over again. Which is what we do not want, because it slows down the process *and* it introduces mistakes all over again. What we need is memory:
# - of previous matches
# - of previous associations
# - of previous wrong associations
# and perhaps more
# 
#     

# In[128]:


def get_delegates_from_spans(ob):
    result = {}
    for s in ob.spans:
        result['delegate'] = s.get_delegate()
        result['pattern'] = s.pattern
    return result


# In[129]:


previously_matched.columns


# In[130]:


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


# In[131]:


abbreviated_delegates.loc[abbreviated_delegates.id==17670]


# In[132]:


framed_gtlm.head()


# In[133]:


len(framed_gtlm.loc[framed_gtlm.ref_id.isin(delresults.keys())])


# In[134]:


[key for key in delresults.keys() if key not in list(framed_gtlm.ref_id)]


# In[135]:


c = Counter()
for x in range(10):
    c.update(chr(x))
c


# In[136]:


get_delegates_from_spans(ob)

ob = searchob.matched_text
for s in ob.spans:
        print(s)
        print(s.get_delegate())
        print(s.idnr)
        print(s.pattern)

# In[137]:


s.__dict__


# In[138]:


mo.search_results


# ## Matching problems
# 
# We cannot resolve this on basis of this isolated keyword matching and there are some more problems  that have to be solved. There are a few things we know that we can use, but the issue is a bit hairy, so let's spell the problems out and then try and formulate a solution to each of them, though this is not strictly separated. 
# 
# - the list of delegates is limited, so if we cannot match a word straight away, we know that the best chance of success is to match it to the list of known delegates. The problem is that at times new delegates will appear, so that the risk of overfitting to a known list of delegates is a good idea.
# - another good idea is to inspect whether a single delegate is not matched twice in a single attendance list. 
#     * This may easily happen with names such as _'Hoorn'_ and _'Hoornbeek'_, especially as Hoornbeek could also be hyphenated.
#     * Another case is _'Taats van Amerongen'_ whose name consists of two parts. The code usually matches both 'Taats' and 'Amerongen' to delegate 'Taats van Amerongen', while he is actually the same person. 
#     * A third and more complicated case is with _'van Heeckeren'_ where there are two 'van Heeckeren's appearing in the same attendance list. They are distinguished by an extension to their name: "van Heeckez ren tot Barlham, van Heeckeren tot den Brandtzenborgh" is "van Heeckeren tot Barlham, van Heeckeren tot den Brandtzenborg". But these are not resolvable using the official names in the delegates database. They are presumably father and son, but there is no way to disambiguate them but by hand.
# - there may be ocr recognition errors in the string we try to match. 
#     * Errors could disappear in a future improved version of the ocr, but there is no way of knowing up front how much this will improve. But still, this may be a reason to leave things as they are for the moment and return to it later.
#     * we could try and ignore or replace characters that we know not to belong in the names, such as digits or f's for long s. But another idea is to use a tailored confusion matrix, that we can make because there are so many instances of this pattern. (and these individual confusion matrices could improve more generalized confusion matrices). It would be a good idea to store these matrices somewhere, though
#     * Something else that frequently happens is that the ocr accidentally separates a name in two (or three or more) parts. 
# - often, strings are part of larger names, that we know of. In identifying we usually strip off the van and tot and other prefixes, but they do belong to the names. For further spotting non-identified strings it would be useful to have as complete a match of all the strings in the attendance list
# - finally: what to do with unidentified (but recurring) delegates, that are not in the database. They tend to fall out now. And we cannot assign them an external id. So i will assign them a separate REPUBLIC id, consisting of a r the date we first found them + a sequence number starting with 1. This means that an unknown delegate found on 3 july 1728 will get id r172807031 and the following r172807032.
# - and
#     
# __Solving strategies__
# 
# - match each and every part of the attendants list as far as we can, including extraordinaris delegates and PRAESIDE etc, so we have no accidentally and implicitly unmatched parts anymore. For this objectwise:
#     * find double matches and try to resolve them by expanding the match strings. See of other nameparts are already matched. If so, join the matched fragments and try to match it with the whole name. We may need to use already gathered  knowledge about appearances of a name. 
#     * match each name to the largest string possible so that we include all the 'vans' and 'vanders' and other prefixes
#     * match all the structural parts such as PRAESIDE
#     * then see how much unmatched names we still have left and try matching them again. If this does not work mark unmatched fragments for later inspection. 
# 
# - develop consistency measures for a year, but also over years: 
#     * which delegates do return
#     * are there any presidents matched that do not recur 
#         * in the same week or as a president elsewhere. Each president has to appear for the first time, so that comparing to neighbouring days is more important.
#         *  If not: are there any likely candidates they should match to, usually in neighbouring days, but perhaps also elseqwhere
#     * are there any delegates that appear very seldom. Could it be that they we recognized wrongly or do their match strings coincide. Are there any likely candidates that they should be matched to anyway?
#     * especially look into the problem of short names, because levenshtein distances are unreliable. Perhaps we need another additional measure for those (ngrams with a 1-ngram for instance)

# In[139]:


ob


# In[140]:


fuzzysearch_config

config = {'char_match_threshold': 0.7,
 'ngram_threshold': 0.6,
 'levenshtein_threshold': 0.6,
 'ignorecase': False,
 'ngram_size': 2,
 'skip_size': 2}

sps = FuzzyKeywordSearcher(config=config)
spns = [(s.pattern, s.get_delegate()) for s in ob.spans if s]
for s in spns:
    nm = s[1]['name']
    d_id = s
#    if nm == 'Taets van Amerongen':
    if not sps.keyword_index.get(nm):
        sps.index_keywords(nm)
# In[141]:


spns


# In[142]:


cand = sps.find_candidates(ob.item)
if cand:
    cand = cand[0]
    span = (cand["match_offset"], cand["match_offset"]+len(cand['match_string']))
    ob.matched_text.set_span(clas='delegate', delegate_id=18838, delegate_name=cand['match_term'], span=span)
    ob.matched_text.get_fragments()


# In[143]:


for key in matches.keys():
    s = score_levenshtein_distance_ratio(key, 'van Glinstra')
    print(key, s)


# # HIER VERDER:
# 
# - Delegates toevoegen
# - beter matchen van niet gevonden patterns in match_unmarked
# - delegates aan overzicht toevoegen
# - datastructures

# In[144]:


ob = searchobs['meeting-1728-05-04-session-1']
fragments = ob.matched_text.get_fragments()
for fragment in fragments:
    print(fragment)


# In[145]:


all_matched.keys()


# In[146]:


spans = ob.matched_text.spans
for span in spans:
    print(span.type, span.idnr, span.pattern, getattr(span, 'delegate_id', 'niks'), getattr(span, 'delegate_name', 'niks'))


# In[ ]:





# In[147]:


type(span)

mh_df['vs'] = mh_df.variants.apply(lambda x: [e[0] for e in x])mh_df.loc[mh_df.vs.apply(lambda x: 'Six' in x)]
# In[148]:


g.variants


# In[149]:


moregentlemen = [MatchHeer(all_matched[d]) for d in all_matched.keys()]


# In[150]:


unclassified = []
nwheren = []
clean_gentlemen = []
for g in moregentlemen:
    matches = defaultdict(list)
    matchkw = g.matchkw
    matchkw = matchkw.strip()
    nwmatches = {}
    if matchkw == '':
        unclassified = g.variants['general'] # we go to the next instance and look into this later
        continue
    for v in g.variants['general']:
        matches[v.match].append(v)
        for key in matches.keys():
            name = g.name
            nwmatches[matchkw] = []
            s = score_levenshtein_distance_ratio(key, matchkw)
            if s > 0.7:
                nwmatches[matchkw].extend(matches[key])
            else:
                if not re.search('\w+', key):
                    #this is an empty key
                    meanscore = mean([score_ngram_overlap_ratio(kw.form, matchkw, 1) for kw in matches[key]])
                    if meanscore > 0.8:
                         nwmatches[matchkw].extend(matches[key])
        #            print('meanscore: ', key, meanscore)

                # is there no strange 'Van' variant in this?
                elif len(key)<5:
                    s2 = score_ngram_overlap_ratio(key, 'van', 1)
                    if s2 > 0.8:
                        meanscore = mean([score_levenshtein_distance_ratio(kw.form, matchkw, 1) for kw in matches[key]])
                        nwmatches[matchkw].extend(matches[key])
                        #print('meanscore: ', key, meanscore)
                    else: 
                        unclassified.extend(matches[key])
                else:
                    meanheerscore = mean([score_levenshtein_distance_ratio(kw.form, matchkw) for kw in matches[key]])
                    meankeyscore = mean([score_levenshtein_distance_ratio(kw.form, key) for kw in matches[key]])
                    meanheergramscore = mean([score_ngram_overlap_ratio(kw.form, matchkw, 2) for kw in matches[key]])
                    meankeygramscore = mean([score_ngram_overlap_ratio(kw.form, key, 2) for kw in matches[key]])
                    if meankeyscore>meanheerscore and meankeygramscore>meanheergramscore:
                        #print(key, meankeyscore-meanheerscore, meankeygramscore-meanheergramscore)
                        nwres = iterative_search(key, year=day, df=abbreviated_delegates)
                        rec = {'id': nwres.id.iat[0],
                               'm_kw': key,
                               'name': nwres.name.iat[0],
                               'variants': [(v.form, v.match, v.score) for v in matches[key]]}
                        nwheer = MatchHeer(rec=rec)
                        nwheren.append(nwheer)
    g.variants['new']=nwmatches
        
                
            


# In[151]:


allvariants = Counter()
for g in moregentlemen:
    allvariants.update(g.variants['general'])
allvariants.most_common()


# ok, so we need something do discriminated them. Perhaps the time of their actual delegate appointments?
# another idea is to try and match names more closely. But we do not implement that here (as I do not think there is much to gain here as yet)

# In[152]:



hs = iterative_search('Heekeren', year=1728,df=abbreviated_delegates)
persoonids = hs.id.unique()
list(persoonids)
sids = [f"{persoonid}" for persoonid in persoonids]


# In[153]:


clbm = DisambiguateDelegate()
clbm.closer_best_match(name='Aerssen', date="1628")


# # Consistency Checks
# ## Not finished

# In[154]:


unmt = [searchobs[so].matched_text.get_unmatched_text() for so in searchobs]


# In[ ]:





# In[155]:


wrds = [k[0].form for k in allvariants.most_common() if k[0] not in ['',None]]
fks.find_close_distance_keywords(wrds)


# In[156]:


for nwh in nwheren:
    if nwh.heerid not in all_matched.keys():
        print(nwh.heerid, nwh.name)


# In[157]:


def joinheer(h1, h2):
    if not h1.heerid == h2.heerid:
        raise ValueError('ids not equal')
    forms1 = set([v.form for form in h1.variants])
    forms2 = set([v.form for form in h2.variants])
    difference = forms2.difference(forms1)
    
    for form in difference:
        v1 = [v for v in h2.variants['general'] if v.form==form][0]
        h1.variants['general'].append(v1)
    return h1


# In[158]:


nnwheren = []


# In[159]:


idvals = set([n.heerid for n in nwheren])
clusterednwh = defaultdict(list)
for heerid in idvals:
    clusterednwh[heerid] = [nwh for nwh in nwheren if nwh.heerid == heerid]
    


# In[160]:


from itertools import groupby
ds = {}
sh = sorted(nwheren, key=lambda x: x.heerid)
for k, g in groupby(sh, key=lambda x: x.heerid):
    ds[k] = list(g)
for k in ds:
    h1 = ds[k][0]
    for h2 in ds[k][1:]:
        h1 = joinheer(h1,h2)
    nnwheren.append(h1)


# In[161]:


nnwheren[2].variants


# In[162]:


unclassified


# In[163]:


unob = unclassified[0]
unob.heer


# In[164]:


c = Counter(unclassified)
c.most_common()


# In[165]:


unclas = fks.find_close_distance_keywords([v.form for v in unclassified])
G_unclas = nx.Graph()
d_nodes = sorted(unclas)
for node in d_nodes:
    attached_nodes = unclas[node]
    G_unclas.add_node(node)
    for nod in attached_nodes:
        G_unclas.add_edge(node, nod)
connected_unclas = list(nx.connected_components(G_unclas))


# In[166]:


connected_unclas


# In[167]:


iterative_search('Ten Brincke', year=1728, df=abbreviated_delegates)


# In[168]:


iterative_search('Eck', year=1728, df=abbreviated_delegates)


# In[141]:


for x in connected_unclas:
    for i in x:
        if not re.search('[0-9]+',i):
            print (i)
#         ir = [v for v in unclassified if v.form == i]
#         for vir in ir:
#             print (vir.form, vir.heerid, '\n\n')
            result = herensearcher.find_candidates(i)
            if len(result) > 0:
                eindresult = max(result, key=lambda keyword: keyword['levenshtein_distance'])
                

[g for g in moregentlemen if score_ngram_overlap_ratio(g.matchkw, 'Taats van Amerongen', 1) > 0.95]
# # Placeholder
iterative_search('Goslinga', year=day, df=abbreviated_delegates)
# In[169]:


framed_gtlm = pd.DataFrame(moregentlemen)


# In[170]:


framed_gtlm.rename(columns={0:'gentleobject'}, inplace=True)

framed_gtlm.to_pickle('sheets/framed_gtlm.pickle')
# In[171]:


framed_gtlm['vs'] = framed_gtlm.gentleobject.apply(lambda x: [e for e in x.variants['general']])
framed_gtlm['ref_id'] = framed_gtlm.gentleobject.apply(lambda x: x.heerid)
framed_gtlm['uuid'] = framed_gtlm.gentleobject.apply(lambda x: x.get_uuid())
framed_gtlm['name'] = framed_gtlm.gentleobject.apply(lambda x: x.name)
framed_gtlm['found_in'] = day
framed_gtlm


# In[172]:


def levenst_vals(x, txt):
    scores= [score_levenshtein_distance_ratio(i.form,txt) for i in x if len(i.form)>1]
    if max(scores) > 0.8:
        return max(scores)
    else:
        return np.nan
    
x = framed_gtlm.iloc[0].vs
txt = 'Lynden'
z = framed_gtlm.vs.apply(lambda x: levenst_vals( x, txt))
#[score_levenshtein_distance_ratio(i.form,txt) for i in x if len(i.form)>1]
framed_gtlm[z==z.max()]
#framed_gtlm.loc[framed_gtlm.vs.apply(lambda x: score_levenshtein_distance_ratio(x, 'Van Glinstra')0.0)]

iterative_search('Cattenborgh', year=1728,df=abbreviated_delegates)z == z.max()framed_gtlm.loc[z==z.max()]
# In[173]:


def span_checker(ob):
    """prints all spans from a searchobject"""
    spans = ob.matched_text.spans
    for span in spans:
        txt = ob.matched_text.item[span.begin:span.end]
        span.set_pattern(txt)

        #kand = matchsearch.find_candidates(txt)[0]
        msk = framed_gtlm.vs.apply(lambda x: levenst_vals(x, txt))
        mres = framed_gtlm.loc[msk==msk.max()]
        if len(mres)>0:

    #         setattr(span, 'delegate_id', mres.uuid.iat[0])
    #         setattr(span, 'delegate_name', mres.name.iat[0])
        #print(kand, all_matched[kand])
            span.set_delegate(delegate_id=mres.ref_id.iat[0], delegate_name=mres.name.iat[0], delegate_score=msk.max())
        print(span.type, span.idnr, txt, span.delegate_id, span.delegate_name, span.delegate_score)

mresfor span in spans:
    print(span.type, span.idnr,  span.pattern, span.delegate_id, span.delegate_name, span.type)
# # Delegates to spans
# 
# should we combine this with finding a better span based on the full name??

# In[174]:


def delegates2spans(searchob):
    spans = searchob.matched_text.spans
    for span in spans:
        txt = searchob.matched_text.item[span.begin:span.end]
        #print(span.type, span.idnr, txt)
        #kand = matchsearch.find_candidates(txt)[0]
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


# In[175]:


for T in searchobs:
    searchob = searchobs[T]
    #mo = MatchAndSpan(ob, junksweeper=junksweeper, previously_matched=previously_matched)
    delegates2spans(searchob)


# In[176]:


zelf = searchob
result = {"metadata":{
                        "inventory_num": zelf.invnr,
                        "meeting_lines": zelf.meeting_lines,
                        "coords": zelf.line_coords,
                        "text": zelf.text,
                        "zittingsdag_id": zelf.id,
                        "url": zelf.make_url()},
                  "spans": {}}#zelf.matched_text.to_dict()}
result


# In[177]:


spansresults = []
for s in searchob.matched_text.spans:
        result = {"fragment":(s.begin, s.end),
                   "class":s.type,
                   "pattern":s.pattern,
                   "delegate_id":s.delegate_id,
                   "delegate_name":s.delegate_name}
        spansresults.append(result)
spansresults


# In[180]:


type(s.delegate_id)


# In[179]:


out=[]
for T in searchobs:
    ob = searchobs[T]
    out.append(ob.to_dict())

jsondelegates.columnsout1['spans'][10].keys()
out1 = out[1]

json.dumps(out1)
# In[181]:


with open('/Users/rikhoekstra/Downloads/1728_pres2.json', 'w') as fp:
    json.dump(obj=out, fp=fp)

abbreviated_delegates.to_json()

# In[182]:


jsondelegates = abbreviated_delegates.loc[abbreviated_delegates.id.isin(previously_matched.id)]


# In[183]:


jsondelegates.columns


# In[185]:


jsondelegates.rename(columns={"h_life": "hypothetical_life", "p_interval":"period_active", "sg": "active_stgen"},
                     inplace=True)


# In[186]:


for lj in ['geboortejaar', 'sterfjaar']:
    jsondelegates[lj] = jsondelegates[lj].dt.strftime("%Y-%m-%d")


# In[187]:


interval = pd.Interval(left=1820,right=1890, closed="both")
interval.left


# In[188]:


for interval in ["hypothetical_life", "period_active"]:
    jsondelegates[interval]=jsondelegates[interval].apply(lambda x: [x.left, x.right])


# In[189]:


jsondelegates['raa_ref_id'] = jsondelegates['id']


# In[167]:


delegate_json = jsondelegates[['id', 'raa_ref_id', 'name', 'geboortejaar', 'sterfjaar', 'colleges',
       'functions', 'active_stgen', 'was_gedeputeerde',
       'period_active', 'hypothetical_life']].to_json(orient="records")


# In[168]:


with open("/Users/rikhoekstra/Downloads/delegats.json", 'w') as fp:
    fp.write(delegate_json)

iterative_search('van Welderen', year=1728,df=abbreviated_delegates)variants_rev = {}
for d in all_matched.keys():
    for variant in variants:
        variants_rev[variant] = {'id':all_matched[d]['id'], 
                                 'name': all_matched[d]['name']}
# In[190]:


from IPython.display import HTML
out=[]
for T in searchobs:
    ob = searchobs[T].matched_text
    url = searchobs[T].make_url()
    ob.mapcolors()
    rest = ob.serialize()
    rest = f"\n<h4>{T}</h4>\n" + rest
    if url:
        rest += f"""<br/><br/><a href='{url}'>link naar {T}-image</a><br/>"""
    out.append(rest)
#out.reverse()
HTML("<br><hr><br>".join(out))


with open(f"/Users/rikhoekstra/Downloads/{T}.html", 'w') as flout:
    flout.write(f"<html><body><h1>{T}</h1>\n")
    flout.write("<br><hr><br>".join(out))
    flout.write("</body></html>")searchob.get_fragments()searchob.matched_text.mapcolors()searchob.matched_text.serialize()
# In[194]:


from IPython.display import HTML
out=[]
for T in searchobs:
    ob = searchobs[T].matched_text
    url = searchobs[T].make_url()
    ob.mapcolors()
    rest = ob.serialize()
    rest = f"""\n<tr><td><strong>{T}</strong></td><td>{rest}</td>"""
    if url:
        rest += f"""<td><a href='{url}'>link naar {T}-image</a></td>"""
    rest += "</tr>"
    out.append(rest)
#out.reverse()
outtable = "".join(out)
HTML(f"<table>{outtable}</table>")


# In[195]:


with open(f"/Users/rikhoekstra/Downloads/{day}_check.html", 'w') as flout:
    flout.write(f"<html><body><h1>results for {day}</h1>\n")
    flout.write(f"<table>{outtable}</table>")
    flout.write("</body></html>")

still_unmarked = finders.make_groslijst(presentielijsten=searchobs)
still_unmarked = (w for w in still_unmarked if w not in real_junk)c = Counter(still_unmarked)
c.most_common()checkable = [w for w in c if  5<c[w]<60]
len(checkable)c_checkable = [w for w in checkable if len(junksweeper.find_candidates(w, include_variants=True))==0]c_checkable
# In[196]:


identified = {w:iterative_search(name=w, year=1727, df=abbreviated_delegates) for w in c_checkable}

iterative_search(name='Heecke', year=1727, df=abbreviated_delegates)ob = searchobs['meeting-1727-03-20-session-1'].matched_text
''.join(ob.get_unmatched_text())unmarked_text.find('Waveren,') + len('Waveren,')ob.set_span(span=(124,132), clas='delegate', pattern='Waveren,', delegate=19871, score=1.0)
# In[ ]:


grrr = [s for s in ob.spans if not set(range(s.begin,s.end)).isdisjoint(set((124,132)))]


# In[ ]:


ob

import types
search_results = {}
ob = searchobs['meeting-1727-03-20-session-1'].matched_text
unmarked_text = ''.join(ob.get_unmatched_text())
splittekst = re.split(pattern="\s",string=unmarked_text)
for s in splittekst:
    if len(s)>2 and s not in real_junk:
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
            passfor s in splittekst:
    if len(s)>2 and s not in real_junk:
        sr = identified.get(s)
        print(s, sr)
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
            pass
# In[ ]:


search_results


# In[199]:


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
                pass


# In[ ]:


search_results


# In[ ]:


n_identified = {w:identified[w] for w in identified if len(w) > 3 and  junksweeper.find_candidates(w, include_variants=True)==[]}


# In[ ]:


identified = {w:identify(name=w, year=1726, window=10, df=abbreviated_delegates) for w in n_identified}


# In[ ]:


#from IPython.core.debugger import set_trace
search_results = {}
ob = searchobs['meeting-1726-01-04-session-1'].matched_text
txt = ob.get_unmatched_text()
maintext = ob.item
matchres = {}
unmarked_text = ''.join(ob.get_unmatched_text())
splittekst = re.split(pattern="\s",string=unmarked_text)
for s in splittekst:
    if s not in real_junk and len(s)>2:
        if junksweeper.find_candidates(s, include_variants=True)==[]:
            mtch = previously_matched.name.apply(lambda x: score_levenshtein_distance_ratio(x,s) > 0.6)
            tussenr = previously_matched.loc[mtch]
            if len(tussenr) > 0:
                tussenr['score'] = tussenr.name.apply(lambda x: score_levenshtein_distance_ratio(x,item))
                matchname = tussenr.loc[tussenr.score == tussenr.score.max()]
                nm = matchname.proposed_delegate.iat[0]
                idnr = matchname.id.iat[0]
                score = tussenr.score.max()
                search_results[s] = {'match_term':nm, 'match_string':s, 'score': score}
            else:
                search_result = matchsearch.find_candidates(s, include_variants=True)
                if len(search_result)>0:
                    search_results[s] = max(search_result, 
                                        key=lambda x: x['levenshtein_distance'])
    else:
        pass
for ri in search_results:
    begin = maintext.find(ri)
    end = begin + len(ri)
    span = (begin, end)
    result = search_results[ri]
    comp = [s for s in ob.spans if not set(range(s.begin,s.end)).isdisjoint(span)]
    if comp != []:
        print(ri, comp)
        change = False
        for cs in comp:
            if not set(range(cs.begin,cs.end)).issubset(span):
                ob.spans.remove(cs)
                ob.set_span(span, clas='delegate')
    else:
        ob.set_span(span, clas='delegate')
            
        
# #        deputyregister[result['match_term']].append(T)
#         ob.set_span(span, clas='delegate')

