
# coding: utf-8

# In[1]:

import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:

qtls_NDVI = np.array([0.1,0.16,0.18,0.21,0.23,0.27,0.35,0.47,0.65])
qtls_TRMM = np.array([1.99,3.18,5.02,7.96,13.34,22.47,36.94,58.43,94.4])
qtls_LST = np.array([13.19,13.59,13.94,14.33,14.66,14.88,15.04,15.21,15.42])

qtls_NDVI_lst = qtls_NDVI.tolist()
qtls_NDVI_lst.insert(0,' ')
qtls_NDVI_lst.append(' ')

qtls_TRMM_lst = qtls_TRMM.tolist()
qtls_TRMM_lst.insert(0,' ')
qtls_TRMM_lst.append(' ')  

qtls_LST_lst = qtls_LST.tolist()
qtls_LST_lst.insert(0,' ')
qtls_LST_lst.append(' ')


# In[2]:

df_allyears_raw = pd.read_csv(r'D:\Downloads\Mattijn@Jia\png\trial_5\data_dataframes_csv//df_allyears_raw.csv', 
                                index_col=0, parse_dates=True)
df_allyears_class = pd.read_csv(r'D:\Downloads\Mattijn@Jia\png\trial_5\data_dataframes_csv//df_allyears_class.csv', 
                                index_col=0, parse_dates=True)


# In[3]:

fig = plt.figure(figsize=(12,48))
df_allyears_raw_grouped = df_allyears_raw.groupby(lambda x: x.month)
for month, data_month in df_allyears_raw_grouped:
    print str(month).zfill(2),

    mean = np.mean(data_month.T,axis=1)
    demeaned = data_month-mean
    evals, evecs = np.linalg.eig(np.cov(demeaned.T))
    
    #evals,evecs are not guaranteed to be ordered
    order = evals.argsort()[::-1] 
    print evals[order]
    print evecs[:,order[0]]
    
    m = data_month.mean(axis=0)    

    e1 = evecs[:,order[0]]
    e1v = evals[order[0]]    
    
    l1 = np.array([([0, e1[0]*e1v]+m[0])[1],
                   ([0, e1[1]*e1v]+m[1])[1],
                   ([0, e1[2]*e1v]+m[2])[1]])
    
    l2 = np.array([([0, e1[0]*e1v]-m[0])[1]*-1,
                   ([0, e1[1]*e1v]-m[1])[1]*-1,
                   ([0, e1[2]*e1v]-m[2])[1]*-1])
    
#     #data_month = data_month.as_matrix()
    mean_NDVI  = np.mean(data_month.NDVI)
    mean_CHIRP = np.mean(data_month.CHIRP)
    mean_LST   = np.mean(data_month.LST)
    center     = [mean_NDVI,mean_CHIRP,mean_LST]    
    
    plot1 = month*3-2
    plot2 = month*3-1    
    plot3 = month*3-0        

#     # --- # --- # --- # --- # --- # ax1 # --- # --- # --- # --- # --- #
    ax1 = plt.subplot(12,3,plot1)
    ax1.scatter(data_month.NDVI,data_month.CHIRP, alpha=0.1, s=400,zorder=3)
    ax1.annotate ('', (l1[0], l1[1]), (l2[0], l2[1]), arrowprops={'arrowstyle':'-','linewidth':2,'color':'k'},)
    
#     ax1.xaxis.set_ticklabels(qtls_NDVI_lst)
#     ax1.xaxis.set_label_text('NDVI 5-quantiles')
#     ax1.set_xlim(0,5)
#     ax1.yaxis.set_ticklabels(qtls_TRMM_lst)
#     ax1.yaxis.set_label_text('CHIRP 5-quantiles (mm)')
#     ax1.set_ylim(0,5)
    ax1.grid(True)
    ax1.set_title('NDVI vs CHIRP')
    ax1.text(0.5,4.5, 'month: '+str(month).zfill(2))
    
    # --- # --- # --- # --- # --- # ax2 # --- # --- # --- # --- # --- #
    ax2 = plt.subplot(12,3,plot2)
    ax2.scatter(data_month.NDVI,data_month.LST, alpha=0.1, s=400)
    ax2.annotate ('', (l1[0], l1[2]), (l2[0], l2[2]), arrowprops={'arrowstyle':'-','linewidth':2,'color':'k'},)
    
#     ax2.xaxis.set_ticklabels(qtls_NDVI_lst)
#     ax2.xaxis.set_label_text('NDVI 5-quantiles')
#     ax2.set_xlim(0,5)
#     ax2.yaxis.set_ticklabels(qtls_LST_lst)
#     ax2.yaxis.set_label_text('LST 5-quantiles (degrees)')
#     ax2.set_ylim(0,5)
    ax2.grid(True)
    ax2.set_title('NDVI vs LST')
#     # --- # --- # --- # --- # --- # ax3 # --- # --- # --- # --- # --- #
    
    ax3 = plt.subplot(12,3,plot3)
    ax3.scatter(data_month.CHIRP,data_month.LST, alpha=0.1, s=400,zorder=3)
    ax3.annotate ('', (l1[1], l1[2]), (l2[1], l2[2]), arrowprops={'arrowstyle':'-','linewidth':2,'color':'k'},)    
    
#     ax3.yaxis.set_ticklabels(qtls_LST_lst)
#     ax3.yaxis.set_label_text('LST 5-quantiles (degrees)')
#     ax3.set_ylim(0,5)
#     ax3.xaxis.set_ticklabels(qtls_TRMM_lst)
#     ax3.xaxis.set_label_text('CHIRP 5-quantiles')
#     ax3.set_xlim(0,5)
    ax3.grid(True)
    ax3.set_title('CHIRP vs LST')
    
plt.tight_layout()  
plt.savefig(r'D:\Downloads\Mattijn@Jia\png\trial_6//pca_allmonths_raw_1.png', dpi=300)
plt.show()


# In[ ]:




# In[ ]:



