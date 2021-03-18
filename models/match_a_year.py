from ..elastic_search_helpers import get_praesentielijsten


presentielijsten = get_presentielijsten(es=es_republic, index='pagexml_meeting', year=1728)
len(presentielijsten['hits']['hits'])


# In[34]:


searchobs = results_to_obs(presentielijsten)