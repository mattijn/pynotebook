
# coding: utf-8

# In[1]:

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
get_ipython().magic(u'matplotlib inline')


# In[2]:

r2_0701 = r'D:\tmp\ndai_out\R2\window7horizon1//NDAI_R2_.tif'
ds0701 = gdal.Open(r2_0701).ReadAsArray()
ds0701 = ds0701[~np.isnan(ds0701)]


# In[3]:

r2_1407 = r'D:\tmp\ndai_out\R2\window14horizon7//NDAI_R2_.tif'
ds1407 = gdal.Open(r2_1407).ReadAsArray()
ds1407 = ds1407[~np.isnan(ds1407)]


# In[4]:

r2_2814 = r'D:\tmp\ndai_out\R2\window28horizon14//NDAI_R2_.tif'
ds2814 = gdal.Open(r2_2814).ReadAsArray()
ds2814 = ds2814[~np.isnan(ds2814)]


# In[22]:

plt.figure(figsize=(8,4))
fig = sns.kdeplot(ds0701, bw = 0.01, shade=True, label='forecast horizon 01 days (forecast window 07 days)')
fig = sns.kdeplot(ds1407, bw = 0.01, shade=True, label='forecast horizon 07 days (forecast window 14 days)')
fig = sns.kdeplot(ds2814, bw = 0.01, shade=True, label='forecast horizon 14 days (forecast window 28 days)')
plt.xlim(0,1)
plt.legend(loc=2)
plt.title('Univariate distribution of $R^2$ values over Inner-Mongolia region')
plt.xlabel('$R^2$ values')
plt.ylabel('Normalised frequency')
plt.savefig(r'D:\tmp\ndai_out\R2//R2_distribution_graph.png', dpi=200)
plt.show()


# In[ ]:



