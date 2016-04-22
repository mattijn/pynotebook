
# coding: utf-8

# In[155]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[295]:

df_VCI = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_VHI_2003_2013.pkl')


# In[296]:

df_LU = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_LANDUSE.pkl')   
df_LU_14 = df_LU.T[df_LU.T == 14].dropna().T
df_GT = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//groundtruth_2003_2013.pkl')
df_PR = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//GT_Province.pkl')


# In[297]:

rs_list = []
gt_list = []
pr_list = []

for i in df_LU_14.columns:
    #print i
    ID = i
    df = pd.concat([df_VCI[ID], df_GT[ID]], axis=1)    
    df.columns = ['di','gt']
    df['prv'] = pd.Series([df_PR[ID].tolist()[0] for x in range(df.shape[0])], index=df.index)
    df.dropna(inplace = True)
    #df.fillna(0, inplace=True)   
    
    # select observations containing 30 or more consecutive drought days
    df1 = df.loc[df.groupby((~(df.index.to_series().diff() ==  pd.Timedelta(1, unit='d'))
                            ).astype(int).cumsum() ).transform(len).iloc[:, 0] >= 30] 
    
    rs_list = rs_list + df1.ix[:,0].tolist()
    gt_list = gt_list + df1.ix[:,1].tolist()    
    pr_list = pr_list + df1.ix[:,2].tolist()
    #plt.scatter(df_concat.ix[:,1],df_concat.ix[:,0])
    #plt.xlim(0,4)
    #plt.xticks([1,2,3],['Light','Medium','Heavy'])
#plt.show()
rs_array = np.array(rs_list)
gt_array = np.array(gt_list)
pr_array = np.array(pr_list)

rs_Series = pd.Series(rs_array)
gt_Series = pd.Series(gt_array)
pr_Series = pd.Series(pr_array)


# In[298]:

df_new = pd.concat([gt_Series, rs_Series,pr_Series], axis=1)
df_new.columns = ['ground-truth','remote-sensing','province']
df_new.sort('ground-truth', inplace = True)


# In[299]:

from collections import Counter
count_pr = Counter(df_new['province'].tolist()).most_common()

pr_unqna_list = []
pr_unqva_list = []
for i in range(len(count_pr)):
    pr_unqna_list.append(count_pr[i][0])
    pr_unqva_list.append(count_pr[i][1])    

df_stations = pd.DataFrame(pr_unqva_list, index=pr_unqna_list)


# In[300]:

df_area = pd.Series([140000,82300,454000,236660,174000,33940,187700,469000,167000,185900,212000,166600,187400,146000,1200000,66400,720000,205600,156700,156000,487700,394000,101800],index=np.unique(df_new['province']))
df_area_stations = pd.concat([df_stations,df_area], join='inner', axis=1)
df_area_stations.columns = ['stations', 'area']
#df_area_stations.head()

df_sel = df_area_stations['stations']/df_area_stations['area']*1000
sel_names = df_sel[df_sel>=1].index


# In[301]:

df_new_sel = df_new.loc[df_new['province'].isin(sel_names)]


# In[307]:

plt.figure(figsize=(10,5))
ax = sns.boxplot((df_new_sel['remote-sensing'])/1000, groupby=df_new_sel['province'], )
for item in ax.get_xticklabels():
    item.set_rotation(45)
title = 'VHI vs CMA'
ylabel = 'VHI'
ax.set_ylim(0,1)
ax.set_title(title)
ax.set_ylabel(ylabel)
plt.tight_layout()
#plt.savefig(r'C:\Users\lenovo\Desktop//'+title+'.png', dpi=200)


# In[222]:




# In[223]:

df_stations.head()


# In[229]:




# In[240]:




# In[242]:

sel_names[0]


# In[156]:

ax = sns.boxplot(df_new['remote-sensing'], groupby=df_new['ground-truth'])
ax.set_xticklabels(['light', 'medium','heavy'])
ax.set_title('NDAI vs CMA')


# In[115]:

df = df_concat[df_concat['gt']>0]


# In[140]:

get_ipython().run_cell_magic(u'timeit', u'', u"row_labels = df.index[(df.index.to_series() - df.index.to_series().shift(2)) == pd.Timedelta(2, unit='d')]\nrows = [x - pd.Timedelta(n, unit='d') for n in range(0,3) for x in row_labels]\nrows = sorted(rows)\ndf1 = df.loc[rows].groupby(df.loc[rows].index).first()")


# In[110]:

idx = pd.DatetimeIndex(['2003-04-10', '2003-04-11', '2003-04-12', '2003-04-13','2003-04-17','2003-05-02', '2003-05-03', '2003-05-04','2003-07-23', '2003-07-24'])
df = pd.DataFrame(np.random.random((10,2)),index=idx)
df


# In[112]:

row_labels = df.index[(df.index.to_series() - df.index.to_series().shift(2)) == pd.Timedelta(2, unit='d')]
rows = [x - pd.Timedelta(n, unit='d') for n in range(0,3) for x in row_labels]
rows = sorted(rows)
df.loc[rows].drop_duplicates()


# In[ ]:



