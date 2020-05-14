
# coding: utf-8

# In[27]:

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:

shp = gpd.read_file(r'D:\Data\NDAI_VHI_GROUNDTRUTH//groundtruth2_2003_2013.shp')


# In[105]:

def autolabel(rects,total):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        #height = heigh / total * 100.
        #print height
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%s' % np.round(height,1),
                ha='center', va='bottom')


# In[111]:

crop = pd.value_counts(shp['Crop'])
crop_pct = (crop / sum(crop)) * 100
idx = np.arange(len(crop.values))
offset = 0.2
colors = []

for value in crop.index:
    if value == 'Winter wheat':
        colors.append('m')
    else:
        colors.append('k')
        
fig, ax = plt.subplots(figsize=(12,5))
bar = ax.bar(idx+offset, crop_pct.values, color=colors)
width = (bar[0].get_width()/2) + offset

ax.set_ylim(0,70)
ax.set_xticks(idx + width)
ax.set_xticklabels(crop.index, rotation='vertical')
ax.grid(axis='y')
ax.set_ylabel('ground reported drought (%)')
ax.set_xlabel('affected crop')
autolabel(bar, sum(crop))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
plt.tight_layout()
plt.savefig(r'D:\Data\NDAI_VHI_GROUNDTRUTH\png//affected_crops.png', dpi=400)
plt.show()


# In[112]:

df_DI_TCI = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_TCI_2003_2013.pkl')


# In[115]:

shp.head()


# In[121]:

shp_ww = shp.loc[shp['Crop']=='Winter wheat']
shp_ww


# In[120]:

shp_ww.loc[shp_ww['Site_ID'] == 58122]


# In[ ]:



