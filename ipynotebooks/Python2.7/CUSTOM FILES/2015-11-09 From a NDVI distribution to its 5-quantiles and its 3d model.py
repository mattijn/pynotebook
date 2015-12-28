# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

# <codecell>

from __future__ import division
import scipy.signal
import os
import nitime.algorithms as tsa
import os
from osgeo import gdal
import scipy.signal
#plt.style.use('ggplot')
import datetime
import sys

# <codecell>

# Topics: line, color, LineCollection, cmap, colorline, codex
'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc
        
    
def clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    if ax is None: 
        ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): 
        spine.set_visible(False) 

# <codecell>

def listall(RootFolder, wildcard=''):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith('.tif')]
    return lists

# <codecell>

def getDatums(files):
    """
    Get dates from list of files
    Typical date is "2010120" year + doy    
    """
    dates = []
    for filename in files:
        year = int(filename[-11:-7])
        days = int(filename[-7:-4])    
        dates.append(datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1))
    return dates

# <codecell>

def getDatumsLST(files):
    """
    Get dates from list of files
    Typical date is "20101012" year + month + day
    """
    dates = []
    for filename in files:
        year = int(filename[-12:-8])
        month = int(filename[-8:-6])
        days = int(filename[-6:-4])    
        dates.append(datetime.datetime(year, month, days))
    return dates

# <codecell>

def getData(files, path_base, rowx, coly, xBlockSize, yBlockSize):
    """
    Input:
    files_ndvi = list_all(folder_ndvi)    
    path_base = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI\RAW2//NDVI_IM_2003001.tif'    
    
    rowx = 300
    coly = 700    
    
    #blockSizes = utils.GetBlockSize(band)
    xBlockSize = 5
    yBlockSize = 5         
    """

    print files[0], path_base, rowx, coly, xBlockSize, yBlockSize
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
            data_array = np.zeros(shape=(len(files),numRows,numCols), dtype=np.float32)

            # select blocks from trmm and ndvi files
            for m in range(len(files)):

                raster = gdal.Open(files[m], gdal.GA_ReadOnly)
                band = raster.GetRasterBand(1)            
                data_array[m] = band.ReadAsArray(j, i, numCols, numRows).astype(np.float)

            # reshape from 3D to 2D
            data_array = data_array.reshape((int(data_array.shape[0]), 
                                             int(data_array.shape[1]*data_array.shape[2]))).T
    return data_array

# <codecell>

def getDataLST(files, path_base):
    """
    Input:
    folder_lst = r'D:\Downloads\Mattijn@Jia\LST_rasdaman'
    path_base = r'D:\Downloads\Mattijn@Jia\LST_rasdaman//LST_20111230.tif'    
    """
    
    print files[0], path_base
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

    # set base array to fill 
    data_array = np.zeros(shape=(len(files),rows,cols), dtype=np.float32)

    for m in range(len(files)):

        raster = gdal.Open(files[m], gdal.GA_ReadOnly)
        band = raster.GetRasterBand(1)            
        data_array[m] = band.ReadAsArray().astype(np.float)

    # reshape from 3D to 2D
    data_array = data_array.reshape((int(data_array.shape[0]), 
                                     int(data_array.shape[1]*data_array.shape[2]))).T
    return data_array

# <codecell>

folder_ndvi = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI\RAW2'
path_base_ndvi = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI\RAW2//NDVI_IM_2003001.tif'    

# row and column to start extracting data
rowx = 300
coly = 700    

# blocksize to select the area of extraction
#blockSizes = utils.GetBlockSize(band)
xBlockSize = 5
yBlockSize = 5 

# get files from list
files_ndvi = listall(folder_ndvi)

# using the parameters do the work
data_NDVI = getData(files_ndvi, path_base_ndvi, rowx, coly, xBlockSize, yBlockSize)

# <codecell>

folder_trmm = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2\30_Day_Period\30_DaySums'
path_base_trmm = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2\30_Day_Period\30_DaySums//TRMM_IM_2003001.tif'

# row and column to start extracting data
rowx = 17
coly = 24    

# blocksize to select the area of extraction
#blockSizes = utils.GetBlockSize(band)
xBlockSize = 2
yBlockSize = 2 

# get files from list
files_trmm = listall(folder_trmm)

# using the parameters do the work
data_TRMM = getData(files_trmm, path_base_trmm, rowx, coly, xBlockSize, yBlockSize)

# <codecell>

folder_lst = r'D:\Downloads\Mattijn@Jia\LST_rasdaman'
path_base_lst = r'D:\Downloads\Mattijn@Jia\LST_rasdaman//LST_20111230.tif'

# get files from list
files_lst = listall(folder_lst)

# using the parameters do the work
data_LST = getDataLST(files_lst, path_base_lst)

# <markdowncell>

# Compute the cubic

# <markdowncell>

# Get quantiles for NDVI

# <codecell>

# mask NDVI where equal to NaN and create pandas time series
ds_ma = np.ma.masked_equal(data_NDVI, data_NDVI.min())
df = pd.Series(ds_ma.compressed())

# <codecell>

# Compute histogram as line plot
y,binEdges=np.histogram(df, bins=100)
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
plt.show()
#plt.savefig(r'D:\Downloads\Mattijn@Jia\png//Histogram_quantiles_NDVI.png', dpi=150)

# <markdowncell>

# For TRMM 30-Day precipitation

# <codecell>

# mask NDVI where equal to NaN and create pandas time series
#ds_ma = np.ma.masked_equal(ap_trmm, ap_ndvi.min())
df = pd.Series(data_TRMM.ravel())

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
plt.hist(df, bins=100,color=plt.rcParams['axes.color_cycle'][0], edgecolor=plt.rcParams['axes.color_cycle'][0])
plt.vlines(qcut_trmm.as_matrix(),0,y.max(), color=plt.rcParams['axes.color_cycle'][3], lw=2,linestyle='solid')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.ylim(0,y.max())
#plt.suptitle('NDVI 5x5 time-series 2003-2012')
plt.title('TRMM 30-day Histogram + 5-Quantiles')
plt.show()
#plt.savefig(r'D:\Downloads\Mattijn@Jia\png//Histogram_quantiles_TRMM30day.png', dpi=150)

# <markdowncell>

# For LST precipitation

# <codecell>

# mask NDVI where equal to NaN and create pandas time series
#ds_ma = np.ma.masked_equal(ap_trmm, ap_ndvi.min())
df = pd.Series(data_LST.ravel()*10)

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

# <markdowncell>

# Get 2009 data

# <codecell>

# NDVI
files_ndvi_2009 = files_ndvi[2191:2556]
dates_ndvi_2009 = getDatums(files_ndvi_2009)

# row and column to start extracting data
rowx = 300
coly = 700    

# blocksize to select the area of extraction
#blockSizes = utils.GetBlockSize(band)
xBlockSize = 5
yBlockSize = 5 

data_NDVI_2009 = getData(files_ndvi_2009, path_base_ndvi, rowx, coly, xBlockSize, yBlockSize)

# <codecell>

# TRMM
files_trmm_2009 = files_trmm[2192:2557]
dates_trmm_2009 = getDatums(files_trmm_2009)

# row and column to start extracting data
rowx = 17
coly = 24    

# blocksize to select the area of extraction
#blockSizes = utils.GetBlockSize(band)
xBlockSize = 2
yBlockSize = 2 

data_TRMM_2009 = getData(files_trmm_2009, path_base_trmm, rowx, coly, xBlockSize, yBlockSize)

# <codecell>

# LST
files_lst_2009 = files_lst[404:449]
dates_lst_2009 = getDatumsLST(files_lst_2009)

# using the parameters do the work
data_LST_2009 = getDataLST(files_lst_2009, path_base_lst)

# <codecell>

# calculate the mean of 2009
avg_lst_2009 = data_LST_2009.mean(axis=0)
avg_ndvi_2009 = data_NDVI_2009.mean(axis=0)
avg_trmm_2009 = data_TRMM_2009.mean(axis=0)

# <codecell>

# create pandas series of the mean and observations
y2009_ndvi = pd.Series(avg_ndvi_2009,dates_ndvi_2009, name='NDVI')
y2009_trmm = pd.Series(avg_trmm_2009,dates_trmm_2009, name='TRMM')
y2009_lst = pd.Series(avg_lst_2009*10,dates_lst_2009, name='LST')

# <codecell>

# concatenate all dimensions and resmaple to decads
y2009_comb = pd.concat([y2009_ndvi,y2009_trmm,y2009_lst],axis=1)
y2009_comb = y2009_comb.resample("M", how='mean')
#y2009_comb.to_excel(r'D:\Downloads\Mattijn@Jia\data//comb_2009_ndvi_trmm_lst_monthly.xlsx')

# <codecell>

print qcut_ndvi.as_matrix().round(2), qcut_lst.as_matrix().round(2), qcut_trmm.as_matrix().round(2)

# <codecell>

# get quantiles for each variable
qtls_NDVI = qcut_ndvi.as_matrix().round(2)
qtls_TRMM = qcut_trmm.as_matrix().round(2)
qtls_LST = qcut_lst.as_matrix().round(2)

# <codecell>

# do a deep copy and classify the variables
y2009_class = y2009_comb.copy()

# <codecell>

# classify NDVI
y2009_class['NDVI'].loc[(y2009_class['NDVI'] > qtls_NDVI[3])] = 4.5
y2009_class['NDVI'].loc[(y2009_class['NDVI'] <= qtls_NDVI[0])] = 0.5
y2009_class['NDVI'].loc[((y2009_class['NDVI'] > qtls_NDVI[0])) & ((y2009_class['NDVI'] <= qtls_NDVI[1]))] = 1.5
y2009_class['NDVI'].loc[((y2009_class['NDVI'] > qtls_NDVI[1])) & ((y2009_class['NDVI'] <= qtls_NDVI[2]))] = 2.5
y2009_class['NDVI'].loc[((y2009_class['NDVI'] > qtls_NDVI[2])) & ((y2009_class['NDVI'] <= qtls_NDVI[3]))] = 3.5

# classify TRMM
y2009_class['TRMM'].loc[(y2009_class['TRMM'] > qtls_TRMM[3])] = 4.5
y2009_class['TRMM'].loc[(y2009_class['TRMM'] <= qtls_TRMM[0])] = 0.5
y2009_class['TRMM'].loc[((y2009_class['TRMM'] > qtls_TRMM[0])) & ((y2009_class['TRMM'] <= qtls_TRMM[1]))] = 1.5
y2009_class['TRMM'].loc[((y2009_class['TRMM'] > qtls_TRMM[1])) & ((y2009_class['TRMM'] <= qtls_TRMM[2]))] = 2.5
y2009_class['TRMM'].loc[((y2009_class['TRMM'] > qtls_TRMM[2])) & ((y2009_class['TRMM'] <= qtls_TRMM[3]))] = 3.5

# classify LST
y2009_class['LST'].loc[(y2009_class['LST'] > qtls_LST[3])] = 4.5
y2009_class['LST'].loc[(y2009_class['LST'] <= qtls_LST[0])] = 0.5
y2009_class['LST'].loc[((y2009_class['LST'] > qtls_LST[0])) & ((y2009_class['LST'] <= qtls_LST[1]))] = 1.5
y2009_class['LST'].loc[((y2009_class['LST'] > qtls_LST[1])) & ((y2009_class['LST'] <= qtls_LST[2]))] = 2.5
y2009_class['LST'].loc[((y2009_class['LST'] > qtls_LST[2])) & ((y2009_class['LST'] <= qtls_LST[3]))] = 3.5

# <codecell>

#%matplotlib inline
%matplotlib qt

# <codecell>

y2009_class

# <codecell>

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

# <codecell>

import matplotlib as mpl

# <codecell>

for ii in np.linspace(0,359,360)[138:139]:
    #CONSTANTS
    NPOINTS = len(y2009_class['NDVI'])
    COLOR='blue'
    #RESFACT=10
    MAP='jet' # choose carefully, or color transitions will not appear smoooth

    # create random data
    np.random.seed(101)
    x = y2009_class['NDVI']
    y = y2009_class['TRMM']
    z = y2009_class['LST']

    %matplotlib inline
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')

    cm = plt.get_cmap(MAP)
    ax.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
    for i in range(NPOINTS-1):
        ax.plot(x[i:i+2],y[i:i+2],z[i:i+2])
    #    ax.scatter(x[i:i+2],y[i:i+2],z[i:i+2])

    #ax.plot(x,y,z)
    ax.scatter(x,y,z,c=np.linspace(0,1,NPOINTS), s=40)


    #ax.text(.05,1.05,'Reg. Res - Color Map')
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)
    ax.set_zlim(0,5)

    qtls_NDVI_lst = qtls_NDVI.tolist()
    qtls_NDVI_lst.insert(0,' ')
    qtls_NDVI_lst.append(' ')

    ax.w_xaxis.set_ticklabels(qtls_NDVI_lst)
    ax.w_xaxis.set_label_text('NDVI 5-quantiles')

    qtls_LST_lst = qtls_LST.tolist()
    qtls_LST_lst.insert(0,' ')
    qtls_LST_lst.append(' ')

    ax.w_yaxis.set_ticklabels(qtls_LST_lst)
    ax.w_yaxis.set_label_text('LST 5-quantiles (degrees)')

    qtls_TRMM_lst = qtls_TRMM.tolist()
    qtls_TRMM_lst.insert(0,' ')
    qtls_TRMM_lst.append(' ')

    ax.w_zaxis.set_ticklabels(qtls_TRMM_lst)
    ax.w_zaxis.set_label_text('TRMM 5-quantiles (mm)')

    ax.azim=int(ii)#ii
    ax.elev=20#ii

    ax1 = fig.add_axes([0.92,0.25,0.03,0.50])
    ax1.text(-8,1.2,'axim: '+str(ax.azim).zfill(3)+' elev: '+str(ax.elev).zfill(3))
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=NPOINTS)

    cb = mpl.colorbar.ColorbarBase(ax1, cmap=cm,
                                    norm=norm,
                                    orientation='vertical')
    #cb.set_label('Some Units')
    cb.ax.set_yticklabels(y2009_class.index.map(lambda x: x.strftime('%d-%m-%Y')))
    #plt.savefig(r'C:\Users\lenovo\Pictures\movie\trial#2//'+str(int(ii))+'.png', dpi=100, bbox_inches='tight')
    plt.show    
    plt.tight_layout
    #plt.clf()

# <codecell>

ax = None
plt.clf()

# <codecell>

np.linspace(0,359,360)

# <codecell>

from matplotlib.collections import LineCollection

# <codecell>

len(y2009_class['NDVI'])

# <codecell>

fig = plt.figure()
ax = plt.axes(projection='3d')
z = np.linspace(0, 1, len(y2009_class['NDVI']))
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)
c = x + y
ax.scatter(x, y, z, c=c)

# <codecell>


# <codecell>


# <codecell>

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(y2009_class['NDVI'],y2009_class['TRMM'],y2009_class['LST'], 'b-')
ax.scatter(y2009_class['NDVI'],y2009_class['TRMM'],y2009_class['LST'], s=40,c=c)

plt.show()

# <codecell>

import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

X = [(0,0,0,1,0),(0,0,1,0,0),(0,1,0,0,0)]
#c = np.linspace(0, 1., num = X.shape[1])[::-1]
#a = np.ones(shape = c.shape[0])
#r = zip(a, c, c, a) # an attempt to make red vary from light to dark
r = [(1.0, 1.0, 1.0, 1.0), (1.0, 0.75, 0.75, 1.0), (1.0, 0.5, 0.5, 1.0), (1.0, 0.25, 0.25, 1.0), (1.0, 0.0, 0.0, 1.0)]
# r, which contains n tuples of the form (r,g,b,a), looks something like this:
# [(1.0, 1.0, 1.0, 1.0), 
# (1.0, 0.99998283232330165, 0.99998283232330165, 1.0),
# (1.0, 0.9999656646466033, 0.9999656646466033, 1.0),
# (1.0, 0.99994849696990495, 0.99994849696990495, 1.0),
# ..., 
# (1.0, 1.7167676698312416e-05, 1.7167676698312416e-05, 1.0),
# (1.0, 0.0, 0.0, 1.0)]

fig = plot.figure()
ax = fig.gca(projection = '3d')


points = np.array([X[0], X[1], X[2]]).T.reshape(-1, 1, 3)
segs = np.concatenate([points[:-1], points[1:]], axis = 1)
#ax.add_collection(Line3DCollection(segs,disc))
ax.add_collection(Line3DCollection(segs, colors=list(r)))
#lc = Line3DCollection(segs, colors = r)
#ax.add_collection3d(lc)

#ax.set_xlim(-0.45, 0.45)
#ax.set_ylim(-0.4, 0.5)
#ax.set_zlim(-0.45, 0.45)

plot.show()

# <codecell>

Line3DCollection(disc)

# <codecell>

matplotlib.collections.LineCollection()

# <codecell>


# <codecell>


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
#fig.savefig(r'D:\Downloads\Mattijn@Jia\png//cubic2.png', dpi=150)

# <codecell>


# <codecell>


# <codecell>


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


