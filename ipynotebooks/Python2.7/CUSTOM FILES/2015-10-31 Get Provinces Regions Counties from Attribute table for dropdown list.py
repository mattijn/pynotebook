# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
import json

# <codecell>

df = pd.read_csv(r'D:\Data\ChinaShapefile\CHN_adm//CHINA_PROVINCE_REGION_COUNTY2.csv')

# <codecell>

df.sort('NAME_1', inplace=True)
df.set_index(['NAME_1','NAME_2','NAME_3'], inplace=True)

# <codecell>

provinces_list=[]
regions_list=[]
counties_list=[]

ix_chn = 0
provinces = np.unique(df.index.get_level_values('NAME_1').values)

province_code = str(ix_chn)
province_names = np.unique(df.index.get_level_values('NAME_1').values).tolist()

# append provinces to its list
provinces_list.append(province_code)
provinces_list.append(province_names)

for ix_pr, pr in enumerate(provinces):
    region_list = []
    #print ix_pr
    #print pr

    df_prov = df.xs(pr, level='NAME_1', drop_level=False)    
    regions = np.unique(df_prov.index.get_level_values('NAME_2').values)
    
    region_code = str(ix_chn)+'_'+str(ix_pr)
    region_name = np.unique(df_prov.index.get_level_values('NAME_2').values).tolist()
    
    region_list.append(region_code)
    region_list.append(region_name)
    
    regions_list.append(region_list)
    
    for ix_rg, rg in enumerate(regions):
        
        county_list = []
        #print ix_rg
        #print rg
        
        df_regions = df.xs(rg, level='NAME_2', drop_level=False)
        
        county_code = str(ix_chn)+'_'+str(ix_pr)+'_'+str(ix_rg)
        county_names = np.unique(df_regions.index.get_level_values('NAME_3').values).tolist()
        
        county_list.append(county_code)
        county_list.append(county_names)
        
        counties_list.append(county_list)
        #print county_list
print provinces_list

# <codecell>

with open(r'D:\GoogleChromeDownloads\MyWebSites\CHINA_DROPDOWN\js\counties_china_v02.txt', 'w') as thefile:
    for item in counties_list:
        thefile.write("%s\n" % json.dumps(item))
    for item in regions_list:
        thefile.write("%s\n" % json.dumps(item))
    for item in provinces_list:
        thefile.write("%s\n" % json.dumps(item))        

# <codecell>


