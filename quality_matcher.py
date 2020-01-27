#!/usr/bin/env python
# coding: utf-8

# In[1622]:


from republic.fuzzy.fuzzy_keyword_searcher import FuzzyKeywordSearcher
from collections import defaultdict
import republic.parser.republic_file_parser as file_parser
from republic.config.republic_config import base_config, set_config_year


from elasticsearch import Elasticsearch
import republic.elastic.republic_elasticsearch as rep_es


# In[1911]:


# we want this in an object so that we can store some metadata on it and retrieve them any time
class TextWithMetadata(object):
    def __init__(self, searchobject):
        metadata = searchobject['_source']['metadata']
        self.text = searchobject['_source'].get('text') or "niks"
        self.meeting_date = metadata.get('meeting_date') or ''
        self.keyword_matches = [k['match_keyword'] for k in metadata.get('keyword_matches')]
        self.para_id = metadata['paragraph_id'] #so that we can later get it back
        self.matched_text = MatchedText(self.text)
        self.matched_text.para_id = self.para_id
        
    def get_meeting_date(self):
        return self.meeting_date
    
    def set_txt(self, txt):
        self.txt = txt
        
    def get_txt(self):
        return self.txt
    
    def __repr__(self):
        return self.text


# In[1699]:


# classes for containers ahead. They will accommodate all sorts of fragments, consisting of a type and a span
# N.B. for debugging purposes they now have fragment ids (to check that fragments are not duplicately displayed)

import itertools
import json
#from collections import namedtuple

def display_in_html(seq):
    """serialize snippets of html"""
    out = ["<div>%s</div><hr>" % x.serialize() for x in seq]
    return "\n\n".join(out)

class TypedFragment(object):
    def __init__(self, fragment=(), t="", pattern="", idnr=0):
        self.type = t
        self.begin = fragment[0]
        self.end = fragment[1]
        self.idnr = idnr
        self.pattern = ""
        
    def __repr__(self):
        return "fragment type {}: {}-{}".format(self.type, self.begin, self.end)
    
    def __lt__(self, other):
        return self.begin<other
    
    def __le__(self, other):
        return self.begin<=other
    
    def __ge__(self, other):
        return self.begin>=other
    
    def __gt__(self, other):
        return self.begin>other
    
    def to_json(self):
        return json.dumps({'fragment':(self.begin, self.end),
                           'class':self.type,
                           'pattern':self.pattern})
    
    def set_pattern(self, pattern):
        self.pattern = pattern


class MatchedText(object):
    nw_id = itertools.count()
    def __init__(self, item=''):
        self.item = item
        self.spans = []
        self.colormap = {}
        self.template = "<div>{}</div><hr>"
        self.mtmpl = """<span style="color:{color}">{txt}</span>"""
        self.para_id = ""
        
    def color_match(self, fragment, color):
        """return marked fragment from item on basis of span"""
        output = provtmp.format(txt=fragment, color=color)
        return output
    
    def mapcolors(self, colors={}):
        self.colormap = colors
    
    def serialize(self):
        """serialize spans into marked item"""
        fragments = self.get_fragments()
        #fragment += "[{}]".format(i.idnr) # for now
        outfragments = []
        for fragment in fragments:
            if fragment[0] != 'unmarked':
                tcolor = self.colormap.get(fragment[0]) or 'unknown'
                ff = self.color_match(fragment[1], color=tcolor)
                outfragments.append(ff)
            else:
                outfragments.append(fragment[1])
        txt = " ".join(outfragments) # may want to turn this in object property
        return txt
    
    def set_span(self, span=(), clas="", pattern=""):
        this_id = next(self.nw_id)
        sp = TypedFragment(fragment=span, t=clas,pattern=pattern, idnr=this_id)
        self.spans.append(sp)
        return this_id
        
    def get_span(self, idnr=None):
        res = [i for i in self.spans if i.idnr==idnr]
        if len(res) > 0:
            return res[0]
        else:
            raise Exception('Error', 'no such fragment')
    
    def get_fragments(self):
        """the text in the form of a list of fragments"""
        end = 0
        fragments = []
        self.spans.sort()
        txt = self.item
        # see if we have overlap if we do send warnings (and display both anyway)
        #for pair in pairwise(self.spans):
        #    overlap = set(pair[0]) & set(pair[1])
        #    if overlap != set():
        #        print(overlap) # this is useful but needs to go elsewhere
        if self.spans != []:
            #startspan = txt[:self.spans[0].begin]
            #fragments.append(('unmarked', startspan))
            for i in self.spans: 
                begin = i.begin
                if end < i.begin-1: # we need the intermediate text as well
                    fragments.append(('unmarked', txt[end:begin]))
                end = i.end
                fragment = (i.type, txt[begin:end])
                fragments.append((fragment))
    #            i2 = self.mtmpl.format(txt=fragment, color=i.type)
    #            i3 = self.item[i.end:]
                end = i.end
        if end < len(txt):
            fragments.append(('unmarked', txt[end:]))
        
        return fragments
    
    
    def get_unmatched_text(self):
        """text represented as unmatched items"""
        fr = self.get_fragments()
        fragments = [fragment[1] for fragment in fr if fragment[0]=='unmarked']
        return fragments
    
    def to_json(self):
        """make json representation"""
        [{"fragment":(s.begin, s.end), "class":s.type, "pattern":s.pattern} for s in self.spans]
        return json.dumps()
    
    def from_json(self, json):
        """construct spans from jsoninput. 
        Json could have an item and fragments with type
        TODO: construct colormap from classes"""
        for item in json:
            self.set_span(item.get('fragment'), item.get("class"), item.get('pattern'))


# In[1623]:


es = Elasticsearch()

config = base_config
year = 1725
data_dir = "/Users/rikhoekstra/surfdrive/rsg_share/data/hocr/1725"
config["year"] = year
config["data_dir"] = data_dir

def get_pages_info(config):
    scan_files = file_parser.get_files(config["data_dir"])
    print("Number of scan files:", len(scan_files))
    return file_parser.gather_page_columns(scan_files)


year_config = set_config_year(config, year, data_dir)


# In[1624]:


persnamen = ['Van Maasdam', 'Cammingha', 'Haersolte', 'Iddekinge', 'Isselmuden', 'van Welderen', 
             'Ockersse', 'Heukelum', 'Rose', 'Rouse', 'van Singendonck', 'Zanders', 'Bentinck', 
             'Taats van Amerongen', 'Emden', 
             'Raadtpenfionaris van Hoornbeeck', 'van Hoorn', 'Van Renswoude', 'Van Dam', 'Torck']


# In[1626]:


fks = FuzzyKeywordSearcher(config=base_config)
fks.index_keywords(persnamen)


# In[1693]:


def set_body(text):

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
                    "must_not":[
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
            "sort": [ ],
           }   

    return body

  


# In[1694]:


set_body('Herman Meyer')


# In[1832]:


pr_results = [TextWithMetadata(ob) for ob in results["hits"]["hits"]]


# In[1862]:


fks2 = FuzzyKeywordSearcher(config=base_config)
fks.index_keywords(terms)


# In[1720]:


terms = ['Heer', 'Secretaris', 'Directeur', 'Ambassadeur', 'Suppliant', 'Envoyez', 'Grave', 'Weduwe', 'Weduwnaar']


# In[1831]:


for inp in terms:
    body = set_body(input)
    results = es.search(index="paragraph_index", body=body) 


# In[1966]:


marked_texts = {}
window = 100
for inp in terms:
    body = set_body(inp)
    results = es.search(index="paragraph_index", body=body) 
    for ob in results["hits"]["hits"]:
        if ob['_source']['metadata']['paragraph_id'] not in marked_texts.keys():
            mt = TextWithMetadata(ob)
            marked_texts[mt.para_id] = mt
        for markedtext in list(marked_texts.values())[:30]:
            mt = markedtext.matched_text
            mt.colormap["keyw_{}".format(inp)] = "purple"
            rest = fks.find_candidates_new(keyword=inp, text=mt.item)
            spans = []
            for res in rest:
                if res != []:
                    txt = mt.item
                    match_string = res['match_string']
                    match_string = match_string.strip()
                    namepattern = re.compile('%s\s?((vander|van)?\s[A-Z]+\w+)' % re.escape(match_string))
                    srch = namepattern.search(txt)
                    if srch:
                        g = srch.groups(0)
                        #print(namepattern, g)
                        ss = srch.span()
                        span = (ofset+ss[0],ss[1]+ofset)
                        if span not in spans:
                            spans.append(span)                
                    else: 
                        break
        for spn in spans:
            mt.set_span(span=spn, clas="keyw")
        



# In[1927]:


namepattern


# In[1965]:


for inp in terms:
        for markedtext in list(marked_texts.values())[:30]:
            mt = markedtext.matched_text
            #mt.colormap["keyw_{}".format(inp)] = "purple"
            rest = fks.find_candidates_new(keyword=inp, text=mt.item)
            spans = []
            for res in rest:
                if res != []:
                    txt = mt.item
                    match_string = res['match_string']
                    match_string = match_string.strip()
                    namepattern = re.compile('%s\s?((vander|van)?\s[A-Z]+\w+)' % re.escape(match_string))
                    srch = namepattern.search(txt)
                    print (mt.para_id, namepattern, srch)


# In[1964]:


txt = marked_texts['year-1725-scan-382-even-para-3']
np = re.compile('Grave\s?((vander|van)?\\s[A-Z]+\\w+)')
np.search(txt.text)


# In[1792]:


for mt in matched_items:
    
    overlaps = []
    for pair in pairwise(mt.spans):
        a=(pair[0].begin, pair[0].end)
        b=(pair[1].begin, pair[1].end)
        overlap = check_overlap(a,b)
        if overlap:
            overlaps.append((a,b))
overlaps


# In[1929]:


len(marked_texts)


# So we found 1302 occurrences of _Heer_ in one volume (probably more occurrences in a single resolution, but we'll solve that later). In many instances Heer will be followed by a name. How to find it.
# how to find a person name:
# - check if the mached term is 'He\[e\]ren', because then there will be more names
# - let's try a regexp checking for capitalized names
# - oh, and also the ubiquitous 'van', 'vander' and 'de'
# 

# In[1711]:


namepattern = re.compile('Heer(en|e)?\s?((vander|van)?\s[A-Z]+\w+)')


# In[1930]:


[{t:namepattern.search(t).groups(0)} for t in marked_texts if namepattern.search(t)]


# In[1745]:


r = []
end = 0
txt = T.text
while end < len(txt):
    srch = namepattern.search(txt)
    if srch:
        g = srch.groups(0)
        print(g, srch.span(), end, txt)
        end = srch.span()[-1]
        txt = txt[end:]
    else: 
        break
        
        


# In[1743]:


namepattern.search("| Heer Ripperda. Piftolen Piftolen. Hofmeetter van fijne Majefteyr 100—|—=  5O—— Secretaris van de Kamer _ j0o_—|_ 25 — Edelman de la Boca —695|_ 2j——— Introduêteur ———= I00—|—— 100% ")


# In[1970]:


def display_in_html(seq):
    """serialize snippets of html"""
    z = list(marked_texts.values())
    lst = [x.matched_text.serialize() for x in z]
    out = ["<div>%s</div><hr>" % t for t in lst]
    return "\n\n".join(out)


# In[1973]:


outext = HTML(display_in_html(marked_texts))


# In[1967]:


z = list(marked_texts.values())[0]
z.matched_text.spans


# In[1942]:


mrk = marked_texts['year-1725-scan-101-odd-para-1']
mrk.matched_text.item


# In[1976]:


with open("marked_text.html", 'w') as of:
    of.write(display_in_html(marked_texts))


# In[ ]:




