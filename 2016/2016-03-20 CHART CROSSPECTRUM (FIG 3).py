
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

df = pd.DataFrame(data=([12,22,10,4,16,9,17]),
                  index=(['Bare areas',
                          'Sparse (<15%) vegetation',
                          "Closed to open (>15%) herbaceous vegetation \n(grassland, savannas or lichens / mosses)",
                          'Open (15-40%) needle-leaved deciduous or evergreen forest (>5m)',
                          'Mosaic vegetation (50-70%) / cropland (20-50%)', 
                          'Mosaic cropland (50-70%) / vegetation (20-50%)', 
                          'Rainfed croplands',
                         ]))


# In[3]:

rects1 = df.plot(kind='barh', legend=None, width=0.90, 
                 color=['#fff5d7', '#ffebaf', '#ffb432', '#286400', '#cdcd66', '#dcf064', 
                        '#ffff64'], 
        grid=False, xticks=None, zorder=4, fontsize=16, figsize=(14.5,4), xlim=(0,23))
plt.grid(axis='x')
plt.tick_params(direction='out', color='white')
plt.xlabel('Fractional abundance (%)', fontsize=16)




def autolabel(rects):
# attach some text labels
    for rect in rects.patches:
        width = int(rect.get_width())
        #print int(width)
        plt.text(rect.get_x()+rect.get_width()-1.0, rect.get_y()+0.15,  '%s'% (width),
                ha='center', va='bottom', zorder=5, fontsize=16)
autolabel(rects1)
rects1.axes.set_visible(True)
rects1.axes.set_frame_on(False)

plt.tight_layout()

pltOut = r'D:\Downloads\Libraries_documents\HOME\Figures paper//out3.png'

#plt.savefig(pltOut, dpi=400)
plt.show()


# In[ ]:



