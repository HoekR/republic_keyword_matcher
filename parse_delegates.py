#!/usr/bin/env python
# coding: utf-8

# # Tagging delegates and spotting them in the corpus

# ## 1 General
# 
# This has some REPUBLIC hocr_parser code


import os
import json



import republic.parser.republic_file_parser as file_parser
from republic.fuzzy.fuzzy_keyword_searcher import FuzzyKeywordSearcher
#

import copy

year = 1725
base_config = {
    "year": year,
    "base_dir": "/Users/rikhoekstra/surfdrive/rsg_share/data/hocr",
    "page_index": "republic_hocr_pages",
    "page_doc_type": "page",
    "tiny_word_width": 15, # pixel width
    "avg_char_width": 20,
    "remove_tiny_words": True,
    "remove_line_numbers": False,
}

def set_config_year(base_config, year):
    config = copy.deepcopy(base_config)
    config["year"] = year
    config["data_dir"] = os.path.join(config["base_dir"], '{}'.format(year))
    return config

def get_pages_info(config):
    scan_files = file_parser.get_files(config["data_dir"])
    print("Number of scan files:", len(scan_files))
    return file_parser.gather_page_columns(scan_files)



year_config = set_config_year(base_config, year)
pages_info = get_pages_info(year_config)


# ## 1a Object definitions

# #### TextWithMetadata

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


# #### TypedFragment

# In[7]:


import itertools


class TypedFragment(object):
    def __init__(self, fragment=(), t="", pattern="", idnr=0):
        self.type = t
        self.begin = fragment[0]
        self.end = fragment[1]
        self.idnr = idnr
        self.pattern = pattern
        self.matched_text = ""
        self.identified_reference = {"reference":"", "provenance": ""}
        
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
    
    def to_dict(self):
        return {'fragment':(self.begin, self.end),
                           'class':self.type,
                           'pattern':self.pattern,
                           'matched_text': self.matched_text,
                           'identified_reference': self.identified_reference or "",
                          }
    def to_json(self):
        return json.dumps(self.to_dict())
    
    def set_pattern(self, pattern):
        self.pattern = pattern
        
    def set_matched_text(self, matched_text):
        self.matched_text = matched_text
        
    def set_reference(self, reference="", provenance=""):
        self.identified_reference = {"reference":reference,
                                     "provenance":provenance}


# test

def test_typed_fragment():
    tf = TypedFragment(fragment=(3,7), idnr=123445, pattern='niks', t="dummy")
    print(tf, tf.to_json())
    tf.set_matched_text('noppes')
    tf.set_reference('woordenboek')
    print(tf, tf.to_json())


# #### MatchedText


class MatchedText(object):
    nw_id = itertools.count()
    def __init__(self, item=''):
        self.item = item
        self.spans = []
        self.para_id = ""
        self.types = []
    
    def set_span(self, span=(), clas="", pattern=""):
        this_id = next(self.nw_id)
        sp = TypedFragment(fragment=span, t=clas,pattern=pattern, idnr=this_id)
        self.spans.append(sp)
        if clas not in self.types:
            self.types.append(clas)
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
        fragments = [s.to_dict() for s in self.spans]
        return json.dumps(fragments)
    
    def from_json(self, json):
        """construct spans from jsoninput. 
        Json could have an item and fragments with type
        TODO: construct colormap from classes"""
        for item in json:
            self.set_span(item.get('fragment'), item.get("class"), item.get('pattern'))


