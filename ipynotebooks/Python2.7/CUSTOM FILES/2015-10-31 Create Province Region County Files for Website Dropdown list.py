# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd

# <codecell>

df = pd.read_csv(r'D:\Data\ChinaShapefile\CHN_adm//CHINA_PROVINCE_REGION_COUNTY.csv')
df.sort('NAME_1', inplace=True)
df.set_index(['NAME_1','NAME_2','NAME_3'], inplace=True)

# <codecell>

for index, row in df.iterrows():
    print row['NAME_1']
    

# <codecell>

for province,new_df in df.groupby(level=2):
    print new_df

# <codecell>

for idate in df.index.get_level_values('NAME_1'):
    print df.ix[idate], idate

# <codecell>


