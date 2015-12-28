
# coding: utf-8

# In[14]:

import json

data =  { 'COVG_NAME_LST_MOD11C2005':'LST_MOD11C2005_uptodate', 
          'COVG_NAME_NDVI_MOD13C1005':'NDVI_MOD13C1005_uptodate',
          'COVG_NAME_PREP_CHIRP_DEKAD':'PREP_CHIRPS2_dekad_uptodate'} 
print 'DATA:', repr(data)

data_string = json.dumps(data)
print 'JSON:', data_string

file_json = r'D:\Downloads\Mattijn@Zhou\coverages_names.json'
with open(file_json, 'w') as outfile:
    json.dump(data, outfile, sort_keys = True, indent = 4, ensure_ascii=False)


# In[15]:

data_load = json.loads(data_string)
#for coverage in data_string:
#    print coverage


# In[16]:

print data_load['COVG_NAME_LST_MOD11C2005']


# In[25]:

def GetCoverageNames():
    file_json = r'D:\Downloads\Mattijn@Zhou\coverages_names.json'
    with open(file_json) as json_data:
        d = json.load(json_data)
    _CoverageID_NDVI = d['COVG_NAME_NDVI_MOD13C1005']
    _CoverageID_LST  = d['COVG_NAME_LST_MOD11C2005']
    return _CoverageID_NDVI, _CoverageID_LST


# In[27]:

CoverageID_NDVI, CoverageID_LST = GetCoverageNames()
print CoverageID_NDVI, CoverageID_LST


# In[23]:

d['COVG_NAME_LST_MOD11C2005']


# In[ ]:



