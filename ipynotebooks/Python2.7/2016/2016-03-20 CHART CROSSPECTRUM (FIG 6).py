
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from __future__ import division
from osgeo import gdal
import numpy as np
import scipy.signal
import os
import matplotlib.pyplot as plt
import nitime.algorithms as tsa


# Define three functions. First the conversion to scale the graph to 10log10 aka decibel. Second the multitaper estimation method to compute the psd, csd, coherence and phase spectrum. Third is the uncertainty of the phase spectrum based on a monte carlo analysis method

# In[2]:

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)

def mtem(x, y):
    """
    multitaper estimation method
    input:
    x  first time series
    y  second time series

    output:
    fkx  power spectral density x
    fky  power spectral density y
    cxy  cross-spectral density xy
    coh  coherence
    ph  phase between xy at input freq
    
    """
    print 'x size', x.shape
    print 'y size', y.shape
    
    # apply multi taper cross spectral density from nitime module
    f, pcsd_est = tsa.multi_taper_csd(np.vstack([x,y]), Fs=1., low_bias=True, adaptive=True, sides='onesided')
    
    # output is MxMxN matrix, extract the psd and csd
    fkx = pcsd_est.diagonal().T[0]
    fky = pcsd_est.diagonal().T[1]
    cxy = pcsd_est.diagonal(+1).T.ravel()
    
    # using complex argument of cxy extract phase component
    ph = np.angle(cxy)
    
    # calculate coherence using csd and psd
    coh = np.abs(cxy)**2 / (fkx * fky)   
    
    return f, fkx, fky, cxy, ph, coh 

def mtem_unct(x_, y_, cf, mc_no=20):
    """
    Uncertainty function using Monte Carlo analysis
    Input:
    x_ = timeseries x
    y_ = timeseries y
    cf = coherence function between x and y
    mc_no = number of iterations default is 20, minimum is 3
    
    Output:
    phif = phase uncertainty bounded between 0 and pi
    """
    print 'iteration no is', mc_no
    
    data = np.vstack([x_,y_])
    # number of iterations
    # flip coherence and horizontal stack    
    cg = np.hstack((cf[:-1], np.flipud(cf[:-1])))
    
    # random time series fx
    mc_fx = np.random.standard_normal(size=(mc_no,len(data[0])))
    mc_fx = mc_fx / np.sum(abs(mc_fx),axis=1)[None].T
    
    # random time series fy
    mc_fy = np.random.standard_normal(size=(mc_no,len(data[0])))
    mc_fy = mc_fy / np.sum(abs(mc_fy),axis=1)[None].T
    
    # create semi random timeseries based on magnitude squared coherence
    # and inverse fourier transform for ys
    ys = np.real(np.fft.ifft(mc_fy * np.sqrt(1 - cg ** 2))) 
    ys = ys + np.real(np.fft.ifft(mc_fx *cg))
    
    # inverse fourier transform for xs
    xs = np.real(np.fft.ifft(mc_fx))
    
    # spectral analysis
    f_s, pcsd_est = tsa.multi_taper_csd(np.vstack([xs,ys]), Fs=1., low_bias=True, adaptive=True, sides='onesided')
    cxyi = pcsd_est.diagonal(+int(xs.shape[0])).T
    phi = np.angle(cxyi)
    
    # sort and average the highest uncertianties
    pl = int(round(0.975*mc_no)+1)
    phi = np.sort(phi,axis=0)        
    phi = phi[((mc_no+1)-pl):pl]
    phi = np.array([phi[pl-2,:],-phi[pl-mc_no,:]])
    phi = phi.mean(axis=0)#
    phi = np.convolve(phi, np.array([1,1,1])/3)
    phif = phi[1:-1]
    return phif


# Extract timeseries from satellite imagery

# In[3]:

import os
from osgeo import gdal
import scipy.signal


# In[4]:

def listall(RootFolder, wildcard=''):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith('.tif')]
    return lists

rowx = 300
coly = 700

#folder_trmm = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\2009\TRMM_10_DaySums_StdNormAnomalyRes_SummerOnly\0'
#folder_ndvi = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\2009\NDVI_DaySums_StdNormAnomaly_SummerOnly\0'
#path_base = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\2009\TRMM_10_DaySums_StdNormAnomalyRes_SummerOnly\0//0_TRMM_IM_2009091.tif'

folder_trmm = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2_2009\10_Day_Period\10_DaySums_StdNormAnomalyRes'
folder_ndvi = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI_2009\DaySums_StdNormAnomaly'
path_base = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI_2009\DaySums_StdNormAnomaly//NDVI_IM_2009001.tif'
#folder12 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\2009\OUTPUT_SummerOnly\0//'


# Read files and check length

# In[5]:

files_trmm = listall(folder_trmm)
files_ndvi = listall(folder_ndvi)
if len(files_ndvi) <> len(files_trmm):
    print 'Input length time series not equal: x ', len(files_ndvi), 'y ', len(files_trmm)
    sys.exit()
else:
    print 'Input length time series equal: series1', len(files_ndvi), 'series2', len(files_trmm)


# Open basefile and set arrays for initial rasters

# In[6]:

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
xBlockSize = 200
yBlockSize = 200

# set initial rasters
fkx_12array = np.zeros_like(base)
fky_12array = np.zeros_like(base)
cxy_12array = np.zeros_like(base)
coh_12array = np.zeros_like(base)
ph_12array = np.zeros_like(base)


# The single loop part for a single timeseries

# In[7]:

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
        ap_ndvi = np.zeros(shape=(len(files_ndvi),numRows,numCols), dtype=np.float32)
        
        # select blocks from trmm and ndvi files
        for m in range(len(files_trmm)):
            raster = gdal.Open(files_trmm[m], gdal.GA_ReadOnly)
            band = raster.GetRasterBand(1)            
            ap_trmm[m] = band.ReadAsArray(j, i, numCols, numRows).astype(np.float)
                
            raster = gdal.Open(files_ndvi[m], gdal.GA_ReadOnly)
            band = raster.GetRasterBand(1)            
            ap_ndvi[m] = band.ReadAsArray(j, i, numCols, numRows).astype(np.float)

        # reshape from 3D to 2D
        ap_trmm = ap_trmm.reshape((int(ap_trmm.shape[0]),int(ap_trmm.shape[1]*ap_trmm.shape[2]))).T
        ap_ndvi = ap_ndvi.reshape((int(ap_ndvi.shape[0]),int(ap_ndvi.shape[1]*ap_ndvi.shape[2]))).T
        
#         # since only the summer/autumn period is used, zero pad to one year data.
#         trmm = np.zeros((ap_trmm.shape[0],364))
#         ndvi = np.zeros((ap_ndvi.shape[0],364))
        
#         # convolution add two values on beginning and end of time series
#         trmm[:,90:335] = scipy.signal.fftconvolve(ap_trmm - ap_trmm.mean(-1)[None].T, np.array(np.ones((1,3))/3)) 
#         ndvi[:,90:335] = scipy.signal.fftconvolve(ap_ndvi - ap_ndvi.mean(-1)[None].T, np.array(np.ones((1,3))/3))
        
#         # extend signal
#         trmm = np.tile(trmm, 200)
#         ndvi = np.tile(ndvi, 200)
        
#         # filter nan values
#         nan_ndvi_ix = np.where(np.mean(ndvi, axis=-1, dtype=np.float64)==nan)
#         nan_trmm_ix = np.where(np.mean(trmm, axis=-1, dtype=np.float64)==nan)
#         trmm[nan_ndvi_ix] = nan        
#         ndvi[nan_trmm_ix] = nan
        
#         sf, fxf, fyf, cxyf, phf, cf = mtem(ndvi,trmm, numRows, numCols) # #fkx_12, fky_12, cxy_12, coh_12, ph_12
        
#         #fky_12array[i:i+numRows, j:j+numCols] = fky_12
#         #fkx_12array[i:i+numRows, j:j+numCols] = fkx_12
#         #cxy_12array[i:i+numRows, j:j+numCols] = cxy_12
#         #coh_12array[i:i+numRows, j:j+numCols] = coh_12
#         #ph_12array[i:i+numRows, j:j+numCols] = ph_12
        

# #path_coher12 = folder12+'coh12.tif'
# #path_phase12 = folder12+'ph12.tif'
# #path_pwsdx12 = folder12+'psdx12.tif'
# #path_pwsdy12 = folder12+'psdy12.tif'
# #path_csdxy12 = folder12+'csdxy12.tif'
        
# #saveRaster(path_coher12, coh_12array, ds, datatype=6, nan=-99999.0)
# #saveRaster(path_phase12, ph_12array, ds, datatype=6, nan=-99999.0)
# #saveRaster(path_pwsdx12, fkx_12array, ds, datatype=6, nan=-99999.0)
# #saveRaster(path_pwsdy12, fky_12array, ds, datatype=6, nan=-99999.0)
# #saveRaster(path_csdxy12, cxy_12array, ds, datatype=6, nan=-99999.0)
# fxf = fxf[0]
# fyf = fyf[0]
# cxyf = cxyf[0]
# phf = phf[0]
# cf = cf[0]


# In[8]:

b = np.ascontiguousarray(ap_trmm).view(np.dtype((np.void, ap_trmm.dtype.itemsize * ap_trmm.shape[1])))
_, idx = np.unique(b, return_index=True)

unique_a = ap_trmm[idx]
unique_b = ap_ndvi[idx]

plt.figure(figsize=(7,5))
for i in unique_a:
    p1, = plt.plot(i, color='c', alpha=0.2)
for j in unique_b:
    p2, = plt.plot(j, color='m', alpha=0.2)

p3, = plt.plot(unique_a.mean(axis=0), 'k-', lw=1.5)
p4, = plt.plot(unique_b.mean(axis=0), 'k--', lw=1.5)


plt.grid(axis='both')
plt.ylim(-0.7,0.5)
plt.xlim(0,364)
plt.xlabel('day of year')
plt.ylabel('normalised anomaly')
plt.title('anomaly time series')
leg = plt.legend([p1,p3,p2,p4],['precipitation','precipitation mean','ndvi','ndvi mean'],ncol=2, loc=3)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_edgecolor('lightgray')
plt.tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME\Figures paper//y_anomaly_timeseries.png', dpi=400)


# In[9]:

m1, m2 = unique_a.mean(axis=0),unique_b.mean(axis=0)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([X.ravel(), Y.ravel()])
# values = np.vstack([m1, m2])
# kernel = stats.gaussian_kde(values)
# Z = np.reshape(kernel(positions).T, X.shape)

# fig = plt.figure(figsize=(14,5))
# ax = fig.add_subplot(111)
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#           extent=[xmin, xmax, ymin, ymax])
# ax.plot(m1, m2, 'k.', markersize=2)
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])
# plt.show()

x = np.tile(m1, 100)
y = np.tile(m2, 100)
t = np.arange(x.shape[0])


# Plot the two timeseries as is

# In[10]:

plt.figure(figsize=(14,5))
plt.subplot(121)
plt.grid()

plt.plot(t,x, 'k-', lw=1, label='x')
plt.plot(t,y, 'k--', lw=1, label='y')
plt.xlim(0,365)

plt.ylabel('normalised anomaly')
plt.xlabel('time (days)')
plt.title('two timeseries')
plt.legend()

plt.subplot(122)
plt.grid()

plt.plot(t,x, 'k-', lw=1, label='x')
plt.plot(t,y, 'k--', lw=1, label='y')
#plt.xlim(0,365)

plt.ylabel('normalised anomaly')
plt.xlabel('time (days)')
plt.title('two timeseries')
plt.legend()
plt.gcf().tight_layout()


# Compute all the components we need: frequencies, power spectral density of x, power spectral density of y, cross spectral density between x and y, coherence between x and y, phase spectrum between x and y and last the phase uncertainty between x and y.

# In[11]:

f, fkx, fky, cxy, ph, coh = mtem(x,y)
phif = mtem_unct(x,y,coh, mc_no=5)


# In[12]:

vspan_start = 0.0026#0.0026
vspan_end = 0.0029#0.0028
xlim_start = 0.000
xlim_end = 0.012


# So now only making some plots of the result. First the psd of x and y.

# In[13]:

plt.figure(figsize=(14,5))
plt.subplot(121)
plt.grid(axis='y')

plt.plot(f,dB(fkx), 'r-', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)
plt.xlim(xlim_start,xlim_end)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)

#plt.ylim(-5,35)

plt.ylabel('power spectrum magnitude ($10log10$)')
plt.xlabel('frequency (Hz)')
plt.title('power spectral density of x')
plt.gcf().tight_layout()

plt.subplot(122)
plt.grid(axis='y')
plt.plot(f,dB(fky), 'b-', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)
plt.xlim(xlim_start,xlim_end)
#plt.ylim(-5,35)

plt.ylabel('power spectrum magnitude ($10log10$)')
plt.xlabel('frequency (Hz)')
plt.title('power spectral density of y')
plt.gcf().tight_layout()


# Timeseries of x contains noise and two components relating to frequency 0.1 and 0.047 Hz. Psd of y only contains one component related to frequency 0.1 Hz. Next we plot the cross spectral density and the corresponding coherence, which is the normalised csd.

# In[14]:

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


# In[15]:

plt.figure(figsize=(5,2.85))
plt.rcParams.update({'axes.labelsize': 'large'})
ax = host_subplot(111, axes_class=AA.Axes)
#ax = fig.add_subplot(121)

# aiaiaiai
ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([-120,50])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

#p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('cross-spectrum TRMM/NDVI', fontsize=25)
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)
# aiaiaiai

ax.grid(axis='y')
ax.plot(f,dB(cxy), 'g-', lw=1)
ax.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)
#ax.set_xlim(xlim_start,xlim_end)
#plt.ylim(-5,35)

ax.set_ylabel('cross-power (10log10)') # regex: ($10log10$)
ax.set_xlabel('frequency (Hz)')
#ax.set_title('cross spectral density of x and y')
ax.set_yscale('symlog')
plt.gcf().tight_layout()
#plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_cross_power.png', dpi=400)


# In[16]:

plt.figure(figsize=(5,2.85))
plt.rcParams.update({'axes.labelsize': 'large'})
ax = host_subplot(111, axes_class=AA.Axes)

# aiaiaiai
ax.set_xlim([xlim_start,xlim_end])

ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

#p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('coherence TRMM/NDVI')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)
# aiaiaiai

# plt.subplot(122)
ax.grid(axis='y')
ax.set_ylim([0,1.01])
ax.plot(f,coh, 'y')

ax.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)
# #plt.axvspan(0.099,0.101, color='gray', alpha=0.1)
ax.set_xlim(xlim_start,xlim_end)
# plt.ylim(0,1.01)

ax.set_ylabel('coherence')
ax.set_xlabel('frequency (Hz)')
#ax.set_title()

plt.gcf().tight_layout()
#plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_coherence.png', dpi=400)


# In[17]:

## PLOT 7
plt.figure(figsize=(5,2.85))
plt.rcParams.update({'axes.labelsize': 'large'})
ax = host_subplot(111, axes_class=AA.Axes)

# aiaiaiai
ax.set_xlim([xlim_start,xlim_end])

ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
#ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
ax.set_yticks([0,1./4*np.pi, np.pi/2, 3./4*np.pi,np.pi])
ax.set_yticklabels([r'$0$', r'$\frac{1}{4}\pi$', r'$\frac{\pi}{2}$', r'$\frac{3}{4}\pi$', 
                    r'$\pi$'])


ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

#p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase uncertainty TRMM/NDVI')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)
#plt.subplot(337)
ax.grid(axis='y')

ax.plot(f,phif, 'c', lw=1)
ax.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)

ax.set_xlim(xlim_start,xlim_end)
ax.set_ylim(-0.05,3.2)

ax.set_ylabel('phase uncert (radian)')
ax.set_xlabel('frequency (Hz)')
#ax.set_title('')

plt.gcf().tight_layout()
#plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_phase_uncertain.png', dpi=400)


# In[18]:

plt.figure(figsize=(5,2.85))
plt.rcParams.update({'axes.labelsize': 'large'})
ax = host_subplot(111, axes_class=AA.Axes)

# aiaiaiai
ax.set_xlim([xlim_start,xlim_end])

ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
#ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticklabels(['$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])


#ax.set_yticks([0,1./4*np.pi, np.pi/2, 3./4*np.pi,np.pi])
#ax.set_yticklabels([r'$0$', r'$\frac{1}{4}\pi$', r'$\frac{\pi}{2}$', r'$\frac{3}{4}\pi$', 
#                    r'$\pi$'])


ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

#p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum TRMM/NDVI + uncertainty ')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)
#plt.subplot(337)
ax.grid(axis='y')

#ax.plot(f,phif, 'c', lw=1)
ax.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)

ax.set_xlim(xlim_start,xlim_end)
#ax.set_ylim(-0.05,3.2)
ax.set_ylim(-np.pi,np.pi)

#ax.set_ylabel('phase uncert (radian)')
ax.set_xlabel('frequency (Hz)')
#ax.set_title('')


#plt.figure(figsize=(14,5))
#plt.subplot(121)
#plt.grid(axis='y')
plt.plot(f,ph, 'm')
#plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)

plt.fill_between(f,ph,(ph-phif),facecolor='c' ,edgecolor='c', lw=0.0)#, where=(ph-phif)>=-np.pi)
plt.fill_between(f,ph,(ph+phif),facecolor='c' ,edgecolor='c', lw=0.0)

plt.fill_between(f,(ph+2*np.pi),((ph+2*np.pi)-phif), where=(ph-phif)<=-np.pi,
                 facecolor='c' ,edgecolor='c', lw=0.0)
plt.fill_between(f,(ph-2*np.pi),((ph-2*np.pi)+phif), where=(ph+phif)>=np.pi,
                 facecolor='c' ,edgecolor='c', lw=0.0)


#plt.xlim(xlim_start,xlim_end)


ax.set_ylabel('phase (radian)')
#ax.set_xlabel('frequency (Hz)')
#plt.title('phase between x and y for all frequencies + uncertainty')
plt.gcf().tight_layout()
#plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_phasespectrum_uncertain.png', dpi=400)


# In[19]:

## PLOT 9
rad2time = ph/(2*np.pi*f)
mtcl2time = phif/(2*np.pi*f)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]


## PLOT 7
plt.figure(figsize=(5,2.85))
plt.rcParams.update({'axes.labelsize': 'large'})
ax = host_subplot(111, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum TRMM/NDVI + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['sync','out-of-sync'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

plt.gcf().tight_layout()
#plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_phase_time.png', dpi=400)


# # COMBI

# In[22]:

plt.figure(figsize=(10,8.55))
plt.rcParams.update({'axes.labelsize': 'large'})

ax = host_subplot(321, axes_class=AA.Axes)

#plt.figure(figsize=(14,5))
#plt.subplot(121)
ax.grid(axis='y')

p1, = ax.plot(t,x, 'k-', lw=1, label='TRMM')
p2, = ax.plot(t,y, 'k--', lw=1, label='NDVI')
ax.set_xlim(0,365)
ax.set_ylim(-0.2,0.2)
ax.set_yticks([-0.2,-0.1,0,0.1,0.2])

ax.set_ylabel('anomaly difference')
ax.set_xlabel('day of year')
ax.set_title('normalised anomaly timeseries')
lg = plt.legend([p1,p2], ['P','NDVI'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)




ax = host_subplot(322, axes_class=AA.Axes)
#ax = fig.add_subplot(121)

# aiaiaiai
ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([-120,50])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
ax.axhline(zorder=2, color='lightgray')#, alpha=1)
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

#p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('CSD P / NDVI', fontsize=25)
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)
# aiaiaiai

ax.grid(axis='y', zorder =1)
ax.plot(f,dB(cxy), 'g-', lw=1, zorder =4)
ax.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35, zorder =3)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)
#ax.set_xlim(xlim_start,xlim_end)
#plt.ylim(-5,35)

ax.set_ylabel('cross-power (10log10)') # regex: ($10log10$)
ax.set_xlabel('frequency (Hz)')
#ax.set_title('cross spectral density of x and y')
ax.set_yscale('symlog')
#plt.gcf().tight_layout()
#plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_cross_power.png', dpi=400)

ax = host_subplot(323, axes_class=AA.Axes)

# aiaiaiai
ax.set_xlim([xlim_start,xlim_end])

ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

#p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('coherence P / NDVI')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)
# aiaiaiai

# plt.subplot(122)
ax.grid(axis='y')
ax.set_ylim([0,1.01])
ax.plot(f,coh, 'y')

ax.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)
# #plt.axvspan(0.099,0.101, color='gray', alpha=0.1)
ax.set_xlim(xlim_start,xlim_end)
# plt.ylim(0,1.01)

ax.set_ylabel('coherence')
ax.set_xlabel('frequency (Hz)')
#ax.set_title()

## PLOT 7
#plt.figure(figsize=(5,2.85))
#plt.rcParams.update({'axes.labelsize': 'large'})
ax = host_subplot(324, axes_class=AA.Axes)

# aiaiaiai
ax.set_xlim([xlim_start,xlim_end])

ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
#ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
ax.set_yticks([0,1./4*np.pi, np.pi/2, 3./4*np.pi,np.pi])
ax.set_yticklabels([r'$0$', r'$\frac{1}{4}\pi$', r'$\frac{\pi}{2}$', r'$\frac{3}{4}\pi$', 
                    r'$\pi$'])


ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

#p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase uncertainty P / NDVI')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)
#plt.subplot(337)
ax.grid(axis='y')

ax.plot(f,phif, 'c', lw=1)
ax.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)

ax.set_xlim(xlim_start,xlim_end)
ax.set_ylim(-0.05,3.2)

ax.set_ylabel('phase uncert (radian)')
ax.set_xlabel('frequency (Hz)')
#ax.set_title('')

#plt.gcf().tight_layout()
#plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_phase_uncertain.png', dpi=400)

#plt.figure(figsize=(5,2.85))
#plt.rcParams.update({'axes.labelsize': 'large'})
ax = host_subplot(325, axes_class=AA.Axes)

# aiaiaiai
ax.set_xlim([xlim_start,xlim_end])

ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
#ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticklabels(['$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])


#ax.set_yticks([0,1./4*np.pi, np.pi/2, 3./4*np.pi,np.pi])
#ax.set_yticklabels([r'$0$', r'$\frac{1}{4}\pi$', r'$\frac{\pi}{2}$', r'$\frac{3}{4}\pi$', 
#                    r'$\pi$'])


ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

#p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum P / NDVI + uncertainty ')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)
#plt.subplot(337)
ax.grid(axis='y')

#ax.plot(f,phif, 'c', lw=1)
ax.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)

ax.set_xlim(xlim_start,xlim_end)
#ax.set_ylim(-0.05,3.2)
ax.set_ylim(-np.pi,np.pi)

#ax.set_ylabel('phase uncert (radian)')
ax.set_xlabel('frequency (Hz)')
#ax.set_title('')


#plt.figure(figsize=(14,5))
#plt.subplot(121)
#plt.grid(axis='y')
plt.plot(f,ph, 'm')
#plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)

plt.fill_between(f,ph,(ph-phif),facecolor='c' ,edgecolor='c', lw=0.0)#, where=(ph-phif)>=-np.pi)
plt.fill_between(f,ph,(ph+phif),facecolor='c' ,edgecolor='c', lw=0.0)

plt.fill_between(f,(ph+2*np.pi),((ph+2*np.pi)-phif), where=(ph-phif)<=-np.pi,
                 facecolor='c' ,edgecolor='c', lw=0.0)
plt.fill_between(f,(ph-2*np.pi),((ph-2*np.pi)+phif), where=(ph+phif)>=np.pi,
                 facecolor='c' ,edgecolor='c', lw=0.0)


#plt.xlim(xlim_start,xlim_end)


ax.set_ylabel('phase (radian)')
#ax.set_xlabel('frequency (Hz)')
#plt.title('phase between x and y for all frequencies + uncertainty')
#plt.gcf().tight_layout()
#plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_phasespectrum_uncertain.png', dpi=400)

## PLOT 9
rad2time = ph/(2*np.pi*f)
mtcl2time = phif/(2*np.pi*f)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]


## PLOT 7
#plt.figure(figsize=(5,2.85))
#plt.rcParams.update({'axes.labelsize': 'large'})
ax = host_subplot(326, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('time-lag(days)')
ax.set_title('t (month)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('time-lag P / NDVI + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['sync','out-of-sync'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

plt.tight_layout()
#plt.gcf().tight_layout()
plt.savefig(r'D:\Downloads\Libraries_documents\HOME\Figures paper//NEW_all.png', dpi=400, pad_inches=0.1)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# Cross spectral density shows single peak for 0.1 Hz and the other peak of 0.047 Hz is still visible but weaker. In the coherence plot its visible there is a clear coherence for the 0.1 Hz peak and the two timeseries are near 0 coherent for the frequency of 0.047 Hz. Next we plot the phase spectrum and phase uncertainty seperately.

# In[ ]:

plt.figure(figsize=(14,5))
plt.subplot(121)
plt.grid(axis='y')

plt.plot(f,ph, 'm', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)
plt.xlim(xlim_start,xlim_end)
plt.ylim(-np.pi,np.pi)

plt.ylabel('phase (radian, bounded [$-{\pi}-{\pi}$])')
plt.xlabel('frequency (Hz)')
plt.title('phase between x and y for all frequencies')
plt.gcf().tight_layout()

plt.subplot(122)
plt.grid(axis='y')

plt.plot(f,phif, 'c', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.35)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)
plt.xlim(xlim_start,xlim_end)
plt.ylim(-0.05,3.2)

plt.ylabel('phase uncertainty (radian, bounded [$0-{\pi}$])')
plt.xlabel('frequency (Hz)')
plt.title('phase uncertainty between x and y for all frequencies')
plt.gcf().tight_layout()


# Phase of frequency 0.1 Hz is stable at 1 radian. Phase uncertainty shows high uncertainty for all frequencies excempt frequency corresponding to 0.1 Hz. So by combining the phase and phase uncertainty we get last two plots with phase in radian and phase in time domain units, which are hours this time.

# In[ ]:

plt.figure(figsize=(14,5))
plt.subplot(121)
plt.grid(axis='y')
plt.plot(f,ph, 'm')
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
#plt.axvspan(0.099,0.101, color='gray', alpha=0.1)

plt.fill_between(f,ph,(ph-phif),facecolor='c' ,edgecolor='c', lw=0.0)#, where=(ph-phif)>=-np.pi)
plt.fill_between(f,ph,(ph+phif),facecolor='c' ,edgecolor='c', lw=0.0)

plt.fill_between(f,(ph+2*np.pi),((ph+2*np.pi)-phif), where=(ph-phif)<=-np.pi,
                 facecolor='c' ,edgecolor='c', lw=0.0)
plt.fill_between(f,(ph-2*np.pi),((ph-2*np.pi)+phif), where=(ph+phif)>=np.pi,
                 facecolor='c' ,edgecolor='c', lw=0.0)


plt.xlim(xlim_start,xlim_end)
plt.ylim(-np.pi,np.pi)

plt.ylabel('phase (radian, bounded [$-{\pi}-{\pi}$])')
plt.xlabel('frequency (Hz)')
plt.title('phase between x and y for all frequencies + uncertainty')
plt.gcf().tight_layout()

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

rad2time = ph/(2*np.pi*f)
mtcl2time = phif/(2*np.pi*f)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]


ax = host_subplot(122, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,366])
#ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (days)')
ax.set_title('time (days)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase between x and y for all frequencies + uncertainty')
ax2.set_xticks([0.05,0.1,0.15,0.2])#0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
ax2.set_xticklabels([str(round(1./0.05, 1)),str(round(1./0.1,1)),str(round(1./0.15,1)),str(round(1./0.2,1))])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)

plt.legend([p2,p3,p4,p1,p5], ['phase-delay','single period','halve period','phase uncertainty estimates', 'frequencies of interest'])
plt.grid(axis='y', zorder=0 )

plt.tight_layout()

plt.show()


# For plot on the right matplotlib shows some glimpses as he found it difficult to plot it perfectly. It's forgiven.
# So to get some results out of the plot. Apply a bandpass filter between 0.099 and 0.101 Hz to see the phase in radian and in hour. 

# In[ ]:

# frequency index
f_lb = vspan_start
f_ub = vspan_end
f_ix = np.where((f > f_lb) * (f < f_ub))[0]
p_r2t = np.mean(rad2time[f_ix], -1)
p_ph = np.mean(ph[f_ix], -1)
print 'phase in radian is', round(p_ph,2)   
print 'which correspond to',round(p_r2t,2), 'days (',(365-round(p_r2t,2))*-1,')'
    


# In[ ]:

#x_  = 4 * np.sin(2*(np.pi/10)*t)
#y_  = 4 * np.sin(2*(np.pi/10)*t+1)
y0  = 0.10 * np.sin(2*(np.pi/(1./0.0027))*t+1.2)
y1  = 0.10 * np.sin(2*(np.pi/(1./0.0027))*t+1.2-p_ph)
#x1 = 4 * np.sin(2*(np.pi/(1./0.1)*t-p_ph))#ps)
#y1 = 4 * np.sin(2*(np.pi/(1./0.1)*t))
plt.figure(figsize=(7,5))
#plt.subplot(121)
plt.grid()

p1, = plt.plot(t,x, 'k-', alpha=0.3,lw=1, label='precipitation mean')
p2, = plt.plot(t,y, 'k--', alpha=0.3, lw=1, label='NDVI mean')

#plt.plot(t,x1, 'r', lw=1, label='x1')
p3, = plt.plot(t,y0, 'c', lw=1.4, label='annual period precipitation')
p4, = plt.plot(t,y1, 'm--', lw=1.4, label='annual period NDVI')
#plt.plot(t,x2, 'b-', lw=1, label='x2')
#plt.plot(t,y1, 'g.', lw=1, label='y1')
#plt.plot(t,y2, 'm-', lw=1, label='y2')

plt.xlim(0,365)

plt.ylabel('normalised anomaly')
plt.xlabel('day of year')
plt.title('anomaly timeseries + annual period')
plt.ylim(-0.25,0.15)
leg = plt.legend([p1,p3,p2,p4],['precipitation mean','annual period precipitation',
                                'NDVI mean', 'annual period NDVI'],ncol=2, loc=3)
leg.get_frame().set_lw(0.5)
leg.get_frame().set_ec('lightgray')
#leg.get_frame().set_alpha(0.5)
plt.tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME\Figures paper//y_anomaly_annual_period', dpi=400)


# That's nice, let's throw everything in one plot. Just for fun

# In[ ]:

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

plt.figure(figsize=(14,7.5))

## PLOT 1
plt.subplot(331)
plt.grid(axis='y')

plt.plot(t,x, 'k-', lw=1, label='x')
plt.plot(t,y, 'k--', lw=1, label='y')

plt.xlim(0,364)
plt.ylim(-0.2,0.2)
plt.yticks([-0.2,-0.1,0,0.1,0.2])
plt.ylabel('normalised anomaly')
plt.xlabel('day of year')
plt.title('anomaly timeseries')
leg = plt.legend(loc=3, frameon=True, ncol=2)
leg.get_frame().set_edgecolor('lightgray')
leg.get_frame().set_lw(0.5)

## PLOT 2
plt.subplot(332)
plt.grid(axis='y')

plt.plot(f,dB(fkx), 'r-', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
#plt.ylim(-5,35)

plt.ylabel('power ($10log10$)')
plt.xlabel('frequency (Hz)')
plt.title('power spectrum x')
#plt.gcf().tight_layout()

## PLOT 3
plt.subplot(333)
plt.grid(axis='y')
plt.plot(f,dB(fky), 'b-', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
#plt.ylim(-5,35)

plt.ylabel('power ($10log10$)')
plt.xlabel('frequency (Hz)')
plt.title('power spectrum y')

## PLOT 4
plt.subplot(334)
plt.grid(axis='y')

plt.plot(f,dB(cxy), 'g-', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
#plt.ylim(-5,35)

plt.ylabel('cross-power ($10log10$)')
plt.xlabel('frequency (Hz)')
plt.title('cross-spectrum x/y')
#plt.gcf().tight_layout()

## PLOT 5
plt.subplot(335)
plt.grid(axis='y')

plt.plot(f,coh, 'y')

plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(0,1.01)

plt.ylabel('coherence')
plt.xlabel('frequency (Hz)')
plt.title('coherence x/y')


## PLOT 6
plt.subplot(336)
plt.grid(axis='y')

plt.plot(f,ph, 'm', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           ['$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(-np.pi,np.pi)

plt.ylabel('phase (radian)')
plt.xlabel('frequency (Hz)')
plt.title('phase spectrum x/y')
#plt.gcf().tight_layout()

## PLOT 7
plt.subplot(337)
plt.grid(axis='y')

plt.plot(f,phif, 'c', lw=1)
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
plt.yticks([0,1./4*np.pi, np.pi/2, 3./4*np.pi,np.pi],
           [r'$0$', r'$\frac{1}{4}\pi$', r'$\frac{\pi}{2}$', r'$\frac{3}{4}\pi$', r'$\pi$'])
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(-0.05,3.2)

plt.ylabel('phase uncert (radian)')
plt.xlabel('frequency (Hz)')
plt.title('phase uncertainty x/y')

## PLOT 8
plt.subplot(338)
plt.grid(axis='y')
plt.plot(f,ph, 'm')

plt.fill_between(f,ph,(ph-phif),facecolor='c' ,edgecolor='c', lw=0.0)#, where=(ph-phif)>=-np.pi)
plt.fill_between(f,ph,(ph+phif),facecolor='c' ,edgecolor='c', lw=0.0)

plt.fill_between(f,(ph+2*np.pi),((ph+2*np.pi)-phif), where=(ph-phif)<=-np.pi,
                 facecolor='c' ,edgecolor='c', lw=0.0)
plt.fill_between(f,(ph-2*np.pi),((ph-2*np.pi)+phif), where=(ph+phif)>=np.pi,
                 facecolor='c' ,edgecolor='c', lw=0.0)

plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1)
plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           ['$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])

plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])


plt.xlim(xlim_start,xlim_end)
plt.ylim(-np.pi,np.pi)

plt.ylabel('phase (radian)')
plt.xlabel('frequency (Hz)')
plt.title('phase spectrum x/y + uncertainty')
#plt.gcf().tight_layout()



## PLOT 9
rad2time = ph/(2*np.pi*f)
mtcl2time = phif/(2*np.pi*f)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]


ax = host_subplot(339, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.1, zorder=-1)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum x/y + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['single period','halve period'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

plt.gcf().tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME\Figures paper//y_psdcsdcohphase.png', dpi=400)

# frequency index
f_lb = vspan_start
f_ub = vspan_end
f_ix = np.where((f > f_lb) * (f < f_ub))[0]
p_r2t = np.mean(rad2time[f_ix], -1)
p_ph = np.mean(ph[f_ix], -1)
print 'phase in radian is', round(p_ph,2)
print 'which correspond to', round(p_r2t,2), 'day'


# In[ ]:



