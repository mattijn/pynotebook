# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

# <codecell>

%matplotlib inline
from __future__ import division
from osgeo import gdal
import numpy as np
import scipy.signal
import os
import matplotlib.pyplot as plt
import nitime.algorithms as tsa
import os
from osgeo import gdal
import scipy.signal
plt.style.use('ggplot')
import datetime

# <codecell>

files_ndvi = listall(folder_ndvi)
files_trmm = listall(folder_trmm)
files_lst = listall(folder_lst)
# select year 2009 for ndvi, trmm and lst
files_ndvi = files_ndvi[2191:2556]
files_trmm = files_trmm[2192:2557]
files_lst = files_lst[404:449]

# <codecell>

dates_ndvi = []
for i in files_ndvi:
    year = int(i[-11:-7])
    days = int(i[-7:-4])    
    dates_ndvi.append(datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1))

# <codecell>

dates_trmm = []
for i in files_trmm:
    year = int(i[-11:-7])
    days = int(i[-7:-4])    
    dates_trmm.append(datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1))

# <codecell>

dates_trmm = []
for i in files_trmm:
    year = int(i[-11:-7])
    days = int(i[-7:-4])    
    dates_trmm.append(datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1))

# <codecell>

dates_lst = []
for i in files_lst:
    year = int(i[-12:-8])
    month = int(i[-8:-6])    
    day = int(i[-6:-4])        
    dates_lst.append(datetime.datetime(year, month, day))

# <codecell>


# <codecell>


# <codecell>

def listall(RootFolder, wildcard=''):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith('.tif')]
    return lists

rowx = 300
coly = 700

#folder_trmm = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2\30_Day_Period\30_DaySums'
folder_ndvi = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI\RAW2'
path_base = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI\RAW2//NDVI_IM_2003001.tif'

# register all of the GDAL drivers
gdal.AllRegister()

# open the image
ds = gdal.Open(path_base, gdal.GA_ReadOnly)
if ds is None:
    print 'Could not open base file'
    sys.exit(1)

# get image size
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

# get the band and block sizes
band = ds.GetRasterBand(1)
base = band.ReadAsArray()
nan = band.GetNoDataValue()

#blockSizes = utils.GetBlockSize(band)
xBlockSize = 5
yBlockSize = 5

#files_ndvi = listall(folder_ndvi)

    
# loop through the rows
for i in range(rowx-1, rowx, yBlockSize): #(0, rows, yBlockSize)
    if i + yBlockSize < rows:
        numRows = yBlockSize
    else:
        numRows = rows - i
    
    # loop through the columns
    for j in range(coly-1, coly, xBlockSize):# (0, cols, xBlockSize)
        if j + xBlockSize < cols:
            numCols = xBlockSize
        else:
            numCols = cols - j
        
        print j, i, numCols, numRows
        # set base array to fill 
        ap_ndvi = np.zeros(shape=(len(files_ndvi),numRows,numCols), dtype=np.float32)
        
        # select blocks from trmm and ndvi files
        for m in range(len(files_ndvi)):
                
            raster = gdal.Open(files_ndvi[m], gdal.GA_ReadOnly)
            band = raster.GetRasterBand(1)            
            ap_ndvi[m] = band.ReadAsArray(j, i, numCols, numRows).astype(np.float)

        # reshape from 3D to 2D
        ap_ndvi = ap_ndvi.reshape((int(ap_ndvi.shape[0]),int(ap_ndvi.shape[1]*ap_ndvi.shape[2]))).T

# <codecell>

def listall(RootFolder, wildcard=''):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith('.tif')]
    return lists

rowx = 17
coly = 24

#folder_trmm = 
folder_trmm = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2\30_Day_Period\30_DaySums'
path_base = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2\30_Day_Period\30_DaySums//TRMM_IM_2003001.tif'

# register all of the GDAL drivers
gdal.AllRegister()

# open the image
ds = gdal.Open(path_base, gdal.GA_ReadOnly)
if ds is None:
    print 'Could not open base file'
    sys.exit(1)

# get image size
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

# get the band and block sizes
band = ds.GetRasterBand(1)
base = band.ReadAsArray()
nan = band.GetNoDataValue()

#blockSizes = utils.GetBlockSize(band)
xBlockSize = 2
yBlockSize = 2

#files_trmm = listall(folder_trmm)
#Input length time series equal: series1', len(files_ndvi), 'series2', len(files_trmm)
    
# loop through the rows
for i in range(rowx-1, rowx, yBlockSize): #(0, rows, yBlockSize)
    if i + yBlockSize < rows:
        numRows = yBlockSize
    else:
        numRows = rows - i
    
    # loop through the columns
    for j in range(coly-1, coly, xBlockSize):# (0, cols, xBlockSize)
        if j + xBlockSize < cols:
            numCols = xBlockSize
        else:
            numCols = cols - j
        
        print j, i, numCols, numRows
        # set base array to fill 
        ap_trmm = np.zeros(shape=(len(files_trmm),numRows,numCols), dtype=np.float32)
        
        # select blocks from trmm and ndvi files
        for m in range(len(files_trmm)):
                
            raster = gdal.Open(files_trmm[m], gdal.GA_ReadOnly)
            band = raster.GetRasterBand(1)            
            ap_trmm[m] = band.ReadAsArray(j, i, numCols, numRows).astype(np.float)

        # reshape from 3D to 2D
        ap_trmm = ap_trmm.reshape((int(ap_trmm.shape[0]),int(ap_trmm.shape[1]*ap_trmm.shape[2]))).T

# <codecell>

def listall(RootFolder, wildcard=''):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith('.tif')]
    return lists

rowx = 17
coly = 24

#folder_trmm = 
folder_lst = r'D:\Downloads\Mattijn@Jia\LST_rasdaman'
path_base = r'D:\Downloads\Mattijn@Jia\LST_rasdaman//LST_20111230.tif'

# register all of the GDAL drivers
gdal.AllRegister()

# open the image
ds = gdal.Open(path_base, gdal.GA_ReadOnly)
if ds is None:
    print 'Could not open base file'
    sys.exit(1)

# get image size
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount

# get the band and block sizes
band = ds.GetRasterBand(1)
base = band.ReadAsArray()
nan = band.GetNoDataValue()

#blockSizes = utils.GetBlockSize(band)
xBlockSize = 2
yBlockSize = 2


#files_lst = listall(folder_lst)

# set base array to fill 
ap_lst = np.zeros(shape=(len(files_lst),rows,cols), dtype=np.float32)

for m in range(len(files_lst)):

    raster = gdal.Open(files_lst[m], gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)            
    ap_lst[m] = band.ReadAsArray().astype(np.float)

# reshape from 3D to 2D

ap_lst = ap_lst.reshape((int(ap_lst.shape[0]),int(ap_lst.shape[1]*ap_lst.shape[2]))).T

# <codecell>

avg_lst = ap_lst.mean(axis=0)
avg_ndvi = ap_ndvi.mean(axis=0)
avg_trmm = ap_trmm.mean(axis=0)

# <codecell>

y2009_ndvi = pd.Series(avg_ndvi,dates_ndvi, name='NDVI')
y2009_trmm = pd.Series(avg_trmm,dates_trmm, name='TRMM')
y2009_lst = pd.Series(avg_lst*10,dates_lst, name='LST')

# <codecell>

y2009_comb = pd.concat([y2009_ndvi,y2009_trmm,y2009_lst],axis=1)
y2009_comb = y2009_comb.resample("M", how='mean')
y2009_comb.to_excel(r'D:\Downloads\Mattijn@Jia\data//comb_2009_ndvi_trmm_lst_monthly.xlsx')

# <codecell>


# <codecell>


# <markdowncell>

# For NDVI

# <codecell>

# mask NDVI where equal to NaN and create pandas time series
ds_ma = np.ma.masked_equal(ap_ndvi, ap_ndvi.min())
df = pd.Series(ds_ma.compressed())

# <codecell>

# Compute histogram as line plot
y,binEdges=np.histogram(df,bins=100)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

# Compute the 5-quantiles using pandas quantile function
qcut_ndvi = df.quantile([0.2,0.4,0.6,0.8])
qcut_ndvi.as_matrix().round(2)

# <codecell>

# Plot normal histogram and add quantiles as vlines
#plt.plot(bincenters,y,lw=1, color=plt.rcParams['axes.color_cycle'][5])
plt.hist(ds_ma.flatten(), bins=100,color=plt.rcParams['axes.color_cycle'][0], edgecolor=plt.rcParams['axes.color_cycle'][0])
plt.vlines(qcut_ndvi.as_matrix(),0,y.max(), color=plt.rcParams['axes.color_cycle'][3], lw=2,linestyle='solid')
plt.xlabel('NDVI')
plt.ylabel('Frequency')
plt.ylim(0,y.max())
#plt.suptitle('NDVI 5x5 time-series 2003-2012')
plt.title('NDVI Histogram + 5-Quantiles')

plt.savefig(r'D:\Downloads\Mattijn@Jia\png//Histogram_quantiles_NDVI.png', dpi=150)

# <markdowncell>

# For TRMM 30-Day precipitation

# <codecell>

# mask NDVI where equal to NaN and create pandas time series
#ds_ma = np.ma.masked_equal(ap_trmm, ap_ndvi.min())
df = pd.Series(ap_trmm.ravel())

# <codecell>

# Compute histogram as line plot
y,binEdges=np.histogram(df,bins=100)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

# Compute the 5-quantiles using pandas quantile function
qcut_trmm = df.quantile([0.2,0.4,0.6,0.8])
qcut_trmm.as_matrix().round(2)

# <codecell>

# Plot normal histogram and add quantiles as vlines
#plt.plot(bincenters,y,lw=1, color=plt.rcParams['axes.color_cycle'][5])
plt.hist(ap_trmm.ravel(), bins=100,color=plt.rcParams['axes.color_cycle'][0], edgecolor=plt.rcParams['axes.color_cycle'][0])
plt.vlines(qcut_trmm.as_matrix(),0,y.max(), color=plt.rcParams['axes.color_cycle'][3], lw=2,linestyle='solid')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.ylim(0,y.max())
#plt.suptitle('NDVI 5x5 time-series 2003-2012')
plt.title('TRMM 30-day Histogram + 5-Quantiles')

plt.savefig(r'D:\Downloads\Mattijn@Jia\png//Histogram_quantiles_TRMM30day.png', dpi=150)

# <markdowncell>

# For LST precipitation

# <codecell>

# mask NDVI where equal to NaN and create pandas time series
#ds_ma = np.ma.masked_equal(ap_trmm, ap_ndvi.min())
df = pd.Series(ap_lst.ravel()*10)

# <codecell>

# Compute histogram as line plot
y,binEdges=np.histogram(df,bins=100)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

# Compute the 5-quantiles using pandas quantile function
qcut_lst = df.quantile([0.2,0.4,0.6,0.8])
qcut_lst.as_matrix().round(2)

# <codecell>

# Plot normal histogram and add quantiles as vlines
#plt.plot(bincenters,y,lw=1, color=plt.rcParams['axes.color_cycle'][5])
plt.hist(df, bins=100,color=plt.rcParams['axes.color_cycle'][0], edgecolor=plt.rcParams['axes.color_cycle'][0])
plt.vlines(qcut_lst.as_matrix(),0,y.max(), color=plt.rcParams['axes.color_cycle'][3], lw=2,linestyle='solid')
plt.xlabel('LST (degrees)')
plt.ylabel('Frequency')
plt.ylim(0,y.max())
#plt.suptitle('NDVI 5x5 time-series 2003-2012')
plt.title('LST Histogram + 5-Quantiles')

plt.savefig(r'D:\Downloads\Mattijn@Jia\png//Histogram_quantiles_LST.png', dpi=150)

# <codecell>

print qcut_ndvi.as_matrix().round(2), qcut_lst.as_matrix().round(2), qcut_trmm.as_matrix().round(2)

# <codecell>

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# <codecell>

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
#fig = plt.figure()
fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

#ax.w_xaxis.line.set_color("red")
#ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#ax.figure.set_facecolor('w')
ax.set_axis_bgcolor('w')
ax.set_xlim(-0.8,0.2)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
        
ax.w_xaxis.set_ticklabels([' ', 0.49, 0.31, 0.19, 0.05, ' '])
ax.w_xaxis.set_label_text('NDVI 5-quantiles')

ax.w_yaxis.set_ticklabels([' ', 12.99,13.98,14.82,15.18, ' '])
ax.w_yaxis.set_label_text('LST 5-quantiles (degrees)')


ax.w_zaxis.set_ticklabels([' ', 3.71,8.02,19.53,52.47, ' '])
ax.w_zaxis.set_label_text('TRMM 5-quantiles (mm)')

# Face 1
x1 = np.array([[0, 0.2, 0.2, 0, 0],
               [0, 0, 0, 0, 0]])
y1 = np.array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]])
z1 = np.array([[0, 0, 0.2, 0.2, 0],
               [0, 0, 0, 0, 0]])
# Face 2
x2 = np.array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]])
y2 = np.array([[0, 0.2, 0.2, 0, 0],
               [0, 0, 0, 0, 0]])
z2 = np.array([[0, 0, 0.2, 0.2, 0],
               [0, 0, 0, 0, 0]])
# Face 3
x3 = np.array([[0, 0.2, 0.2, 0, 0],
               [0, 0, 0, 0, 0]])
y3 = np.array([[0, 0, 0.2, 0.2, 0],
               [0, 0, 0, 0, 0]])
z3 = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
               [0.2, 0.2, 0.2, 0.2, 0.2]])
# Face 4
x4 = np.array([[0, 0.2, 0.2, 0, 0],
               [0, 0, 0, 0, 0]])
y4 = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
               [0.2, 0.2, 0.2, 0.2, 0.2]])
z4 = np.array([[0, 0, 0.2, 0.2, 0],
               [0, 0, 0, 0, 0]])
# Face 5
x5 = np.array([[0, 0, 0.2, 0.2, 0],
               [0, 0, 0, 0, 0]])
y5 = np.array([[0, 0.2, 0.2, 0, 0],
               [0, 0, 0, 0, 0]])
z5 = np.array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]])
# Face 6
x6 = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
               [0.2, 0.2, 0.2, 0.2, 0.2]])
y6 = np.array([[0, 0.2, 0.2, 0, 0],
               [0, 0, 0, 0, 0]])
z6 = np.array([[0, 0, 0.2, 0.2, 0],
               [0, 0, 0, 0, 0]])


#ax.plot_surface(x1,y1,z1)
#ax.plot_surface(x2,y2,z2)
#ax.plot_surface(x3,y3,z3)
#ax.plot_surface(x4,y4,z4)
#ax.plot_surface(x5,y5,z5)
#ax.plot_surface(x6,y6,z6)
ax.scatter([0.1],[0.1],[0.1],c="pink",s=100)#jan
ax.scatter([0.1],[0.3],[0.1],c="magenta",s=100)#feb
ax.scatter([0.1],[0.3],[0.1],c="mediumorchid",s=100)#march
ax.scatter([-0.3],[0.5],[0.5],c="slateblue",s=100)#april
ax.scatter([-0.5],[0.7],[0.7],c="dodgerblue",s=100)#may
ax.scatter([-0.7],[0.9],[0.7],c="cadetblue",s=100)#june
ax.scatter([-0.7],[0.7],[0.9],c="darkslategray",s=100)#july
ax.scatter([-0.7],[0.9],[0.7],c="mediumaquamarine",s=100)#august
ax.scatter([-0.5],[0.7],[0.7],c="green",s=100)#september
ax.scatter([-0.3],[0.5],[0.5],c="darksage",s=100)#october
ax.scatter([-0.1],[0.3],[0.3],c="olivedrab",s=100)#november
ax.scatter([0.1],[0.1],[0.3],c="darkkhaki",s=100)#december

fig.tight_layout()
fig.savefig(r'D:\Downloads\Mattijn@Jia\png//cubic2.png', dpi=150)

# <codecell>

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

points = np.array([(1,1,1), (2,2,2)])
labels = ['billy', 'bobby']

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
xs, ys, zs = np.split(points, 3, axis=1)
sc = ax.scatter(xs,ys,zs)

# if this code is placed inside a function, then
# we must use a predefined global variable so that
# the update function has access to it. I'm not
# sure why update_positions() doesn't get access
# to its enclosing scope in this case.
global labels_and_points
labels_and_points = []

for txt, x, y, z in zip(labels, xs, ys, zs):
    x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
    label = plt.annotate(
        txt, xy = (x2, y2), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    labels_and_points.append((label, x, y, z))


def update_position(e):
    for label, x, y, z in labels_and_points:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        label.xy = x2,y2
        label.update_positions(fig.canvas.renderer)
    fig.canvas.draw()

fig.canvas.mpl_connect('motion_notify_event', update_position)

plt.show()

# <codecell>


