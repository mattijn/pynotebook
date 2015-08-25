# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np

# <codecell>

path = r'D:\Downloads\LiZhangsheng@Mattijn//soil_moisture.csv'
ds = pd.read_csv(path, index_col=0)

# <codecell>

# return top 5 rows of the csv file
ds.head(5)

# <codecell>

# return only the values
ds.values

# <codecell>

# return only the index
ds.index

# <codecell>

# return first item of index
ds.index[0]

# <codecell>

# return only the column
ds.columns

# <codecell>

# return first item of column
ds.columns[0]

# <codecell>

# return last item of column
ds.columns[-1]

# <codecell>

# return  2nd and 3rd item of column
ds.columns[1:3]

# <codecell>

# style/themes for charts available
print plt.style.available

# <codecell>

# set a theme for the plot
plt.style.use('fivethirtyeight')

# <codecell>

# standard plot of csv file using pandas .plot()
ds.plot(figsize=(13,10))

# <codecell>

# custom plot using matplotlib using plt.plot()
fig = plt.figure(figsize=(13,10))

ax = fig.add_subplot(111)

# plot each line separate and style separate, ls means linestyle, lw means linewidth, .T means transpose
plt.plot(ds.T.values[0],ds.index, color='m', ls='-', label=ds.columns[0])
plt.plot(ds.T.values[1],ds.index, color='m', ls='--', label=ds.columns[1])
plt.plot(ds.T.values[2],ds.index, color='m', ls='-', lw=2, label=ds.columns[2])
plt.plot(ds.T.values[3],ds.index, color='c', ls='-', label=ds.columns[3])
plt.plot(ds.T.values[4],ds.index, color='c', ls='--', label=ds.columns[4])
plt.plot(ds.T.values[5],ds.index, color='c', ls='-', lw=2, label=ds.columns[5])
plt.plot(ds.T.values[6],ds.index, color='g', ls='-', label=ds.columns[6])
plt.plot(ds.T.values[7],ds.index, color='g', ls='--', label=ds.columns[7])
plt.plot(ds.T.values[8],ds.index, color='g', ls='-', lw=2, label=ds.columns[8])
plt.plot(ds.T.values[9],ds.index, color='r', ls='-', label=ds.columns[9])
plt.plot(ds.T.values[10],ds.index, color='r', ls='--', label=ds.columns[10])
plt.plot(ds.T.values[11],ds.index, color='r', ls='-', lw=2, label=ds.columns[11])
plt.plot(ds.T.values[12],ds.index, color='y', ls='-', label=ds.columns[12])

# add legend
plt.legend()

# invert y axis 
ax.invert_yaxis()
# label y-axis
ax.set_ylabel('Depth (cm)')

# put ticks from xaxis on top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 
# label x-axis
ax.set_xlabel('Relative soilmoisture (%)')

# dirty hack to place legend within the frame of the plot, done by resizing the axes
box = ax.get_position()

# Put a legend on the right
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Put a legend on the bottom
#ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=5)

# save file if needed
#plt.savefig(r'D:\Downloads\LiZhangsheng@Mattijn//relative_soil.png', dpi=400)

# just show file 
plt.show()

# <codecell>

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# <codecell>

# standard functions to create linear segmented colormap
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
c = mcolors.ColorConverter().to_rgb

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

# <codecell>

# we use 9 different hex colors (see next line), dive between 0 and 1
np.linspace(0,1,9)

# <codecell>

# here we create a continous colorbar using 9 different hex colors
cmap_ = make_colormap([c('#2C2E85'), c('#3954A5'),0.125, c('#3954A5'), c('#0186CB'),0.25, c('#0186CB'), c('#01B3C1'),0.375, 
                       c('#01B3C1'), c('#6CBF4B'),0.5, c('#6CBF4B'), c('#FFF001'),0.625, c('#FFF001'), c('#F37028'),0.75,
                       c('#F37028'), c('#EE252B'),0.875, c('#EE252B'), c('#971A1E')])

# create contour levels 
contour_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110,120]
cont_levels_cb = np.linspace(0,120,1000)

# transform the continous colorbar to have same stepsize as number of contour_levels
colors = cmap_(np.linspace(0, 1, len(contour_levels)))

# <codecell>

fig = plt.figure(figsize=(13,10))

# create single subplot to get access to the axes using ax.
ax = fig.add_subplot(111)

# plot the values (flipped upside down) using filled contours function (contourf)
im = plt.contourf(np.flipud(ds.values), levels=contour_levels,colors=colors)#cmap=cmap_, levels=cont_levels_cb)
#plt.contour(np.flipud(ds.values), levels=contour_levels_, lw=2, colors='k')
# add colorbar for the filled contours
cb = plt.colorbar(im)
# set label text and adjust the size
cb.set_label('Relative soilmoisture (%)', fontsize=20)
# to change the fontsize of the colorbar ticks need to use for loop
for t in cb.ax.get_yticklabels():
     t.set_fontsize(20)

# invert the y axis        
ax.invert_yaxis()
# because we invert the y axis also need the reverse the y axis tick labels, done this by ds.index[::-1]
# compare in other cell ds.index[::-1] vs ds.index. Its reverse. Also adapt fontsize
ax.set_yticklabels(ds.index[::-1], fontsize=20)

# set y axis label plus fontsize
ax.set_ylabel('Depth (cm)', fontsize=20)

# prevous plot the x axis was plotted on the top of chart, now not necesseary
#ax.xaxis.tick_top()
#ax.set_xlabel('Relative soilmoisture (%)')
#ax.xaxis.set_label_position('top') 

# set number of ticks on x axis
ax.set_xticks(np.arange(len(ds.columns)))
# set the labels manually for the x axis, and adjust fontsize,
ax.set_xticklabels(['J','A','S','O','N','D','J','F','M','A','M','J','J'], rotation=0, fontsize=20)
# set the label for the x axis and adjust fontsize
ax.set_xlabel('Month', fontsize=20)


ax.vlines(6,0,7, lw=2.5, color='k')

ax.text(0.44, 0.02,'2013', ha='center', va='center', transform=ax.transAxes, fontsize=20)
ax.text(0.56, 0.02,'2014', ha='center', va='center', transform=ax.transAxes, fontsize=20)
# save file if needed
#plt.savefig(r'D:\Downloads\LiZhangsheng@Mattijn//relative_soil_map.png', dpi=400)
# show the shit
plt.show()

# <codecell>

ds.values.shape

# <codecell>


