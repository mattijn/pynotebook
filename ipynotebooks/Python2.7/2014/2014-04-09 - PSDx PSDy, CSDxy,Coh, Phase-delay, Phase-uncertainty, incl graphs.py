
# coding: utf-8

# In[1]:

import pymat2
import os
import pandas as pd
from osgeo import gdal
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.signal import detrend as sp_detrend
import matplotlib.gridspec as gridspec
#import matplotlib.pyplot as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats.distributions as dist
import libtfr

import nitime.algorithms as tsa
import nitime.utils as utils
from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate

import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import sys, os, re
from scipy import stats
get_ipython().magic(u'matplotlib inline')


# In[ ]:

E, V = libtfr.dpss(365, 8, 4)


# In[ ]:

int(E.shape[0])


# In[ ]:

for i in range(int(E.shape[0])):
    plt.plot(E[i], label="K="+str(i))
    plt.xlim([0,365])
    plt.xlabel("days")
plt.title('Discrete prolate speroidal sequence data tapers')
plt.legend()
#savefig(r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM//dpss.png', dpi=200)


# In[ ]:

def saveRaster(path, array, dsSource, datatype=3, formatraster="GTiff", nan=None): 
    """
    Datatypes:
    unknown = 0
    byte = 1
    unsigned int16 = 2
    signed int16 = 3
    unsigned int32 = 4
    signed int32 = 5
    float32 = 6
    float64 = 7
    complex int16 = 8
    complex int32 = 9
    complex float32 = 10
    complex float64 = 11
    float32 = 6, 
    signed int = 3
    
    Formatraster:
    GeoTIFF = GTiff
    Erdas = HFA (output = .img)
    OGC web map service = WMS
    png = PNG
    """
    # Set Driver
    format_ = formatraster #save as format
    driver = gdal.GetDriverByName( format_ )
    driver.Register()
    
    # Set Metadata for Raster output
    cols = dsSource.RasterXSize
    rows = dsSource.RasterYSize
    bands = dsSource.RasterCount
    datatype = datatype#band.DataType
    
    # Set Projection for Raster
    outDataset = driver.Create(path, cols, rows, bands, datatype)
    geoTransform = dsSource.GetGeoTransform()
    outDataset.SetGeoTransform(geoTransform)
    proj = dsSource.GetProjection()
    outDataset.SetProjection(proj)
    
    # Write output to band 1 of new Raster and write NaN value
    outBand = outDataset.GetRasterBand(1)
    if nan != None:
        outBand.SetNoDataValue(nan)
    outBand.WriteArray(array) #save input array
    #outBand.WriteArray(dem)
    
    # Close and finalise newly created Raster
    #F_M01 = None
    outBand = None
    proj = None
    geoTransform = None
    outDataset = None
    driver = None
    datatype = None
    bands = None
    rows = None
    cols = None
    driver = None
    array = None


# In[ ]:

# Colors via http://colorbrewer2.com/
WHITE, LIGHT_GRAY, GRAY, BLACK = [ "#FFFFFF", "#E5E5E5", "#777777", "#000000" ] 
COLORS = [ "#2C7BB6", "#FF6600", "#660099", "#D7191C", "#FFFFBF", "#ABD9E9"]
BLUE, ORANGE, PURPLE, RED, YELLOW, LIGHT_BLUE = COLORS
HIGH_ALPHA = 0.9
MEDIUM_ALPHA = 0.5
LOW_ALPHA = 0.1
rgba = mpl.colors.colorConverter.to_rgba

def set_styles(style_dict):
    """Set matplotlib styles from nested a nested dictionary"""
    for obj in style_dict: mpl.rc(obj, **style_dict[obj]) 

set_styles({
    "figure": { "figsize": [ 12, 4 ], "facecolor": WHITE },
    "savefig": { "dpi": 200, "bbox": "tight" },
    "patch": { "linewidth": 0.5, "facecolor": ORANGE, "edgecolor": WHITE, "antialiased": True },
    "font": { "size": 8 },
    "legend": { "fontsize": 10 },
    "axes": { 
        "facecolor": LIGHT_GRAY, 
        "edgecolor": WHITE, 
        "linewidth": 1, 
        "grid": True, 
        "titlesize": "large", 
        "labelsize": "large", 
        "labelcolor": BLACK,
        "axisbelow": True,
        "color_cycle": COLORS
    },
    "xtick": { "color": GRAY, "direction": "out" }, #GRAY
    "ytick": { "color": GRAY, "direction": "out" }, #GRAY
    "grid": { "color": WHITE, "linestyle": "-" }
})


# In[ ]:

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


# In[21]:

def listall(RootFolder, wildcard=''):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith('.tif')]
    return lists

def getvalues(rootpath, pc, pr, wildcard=''):
    
    li = []
    da = []
    i = 0
    
    #path = r'D:\Data\GPCP_30acc_AnnualAcc'
    test = listall(rootpath, wildcard)
    #test = test[0:-46]
    #print (len(test))
    #print (test[0])
    for t in test:    
        raster = gdal.Open(t, gdal.GA_ReadOnly)
        nameRas = raster.GetDescription()[-11:-4]
        date = datetime.datetime(int(nameRas[0:4]), 1, 1) + datetime.timedelta(int(nameRas[4:9]) - 1) # replaced "int(nameRas[0:4])" with "2010" first year then add doy
        # print date
        da.append(date)
    
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()    
        
        # print array[879,4142] # Grassland pixel
        try:
            li.append(array[pr, pc])
        except:
            li.append(np.nan)
        
        array = None
        band = None
        raster = None
        #print (str(i))
        i = i + 1
        
    signal = pd.Series(data = li, index = da)
    return signal


# In[ ]:

def cohbias(v,cb):
    if v < 2:
        print 'Warning: degrees of freedom must be greater or equal to two'        
    if cb.all() <0 or cb.all() > 1 :
        print 'Warning: biased coherence should be between zero and one'        
    if v > 50:
        print 'Using 50 degrees of freedom'
        v = 50
    path_cb = r'C:\Users\lenovo\Downloads\Archives\smop-master\smop-master\smop//cohbias'
    lm = loadmat(path_cb, )
    expect = lm.get('expect')
    n = lm.get('n')
    c = lm.get('c')
    #print c.shape, n.shape, expect.shape
    
    cb=cb[:]
    c=c.ravel()
    n=n.ravel()
    v=v
    
    ec=np.zeros_like(c)
    for ct in xrange(len(c)):    
        f = inter.interp1d(n,expect[:,ct], kind='linear')
        ec[ct] = f(v)
        
    cu = np.zeros_like(cb)    
    f = inter.interp1d(ec,c, kind='linear')#c,ec
    for ct in xrange(len(cb)):      
        try:
            cu[ct] = f(cb[ct])
        except ValueError:
            cu[ct] = np.nan
    
    pl = np.where(isnan(cu) & (cb >=0) & (cb < 1))
    cu[pl]=0
    return cu


# In[ ]:

from __future__ import division
from libtfr import dpss
from scipy.io import loadmat,savemat
import scipy.interpolate as inter

def cmtm(x,y,dt=1,NW=8,qbias=0,confn=0):

    """
    Multi-taper method coherence using adaptive weighting and correcting for the bias inherent to coherence estimates.  
    The 95 coherence confidence level is computed by cohconf.m.  In addition, a built-in Monte Carlo estimation procedure 
    is available to estimate phase 95 confidence limits. 

    Inputs:
    x     - Input data vector 1.  
    y     - Input data vector 2.  
    dt    - Sampling interval (default 1) 
    NW    - Number of windows to use (default 8) 
    qbias - Correct coherence estimate for bias (yes, 1)  (no, 0, default).
    confn - Number of iterations to use in estimating phase uncertainty using a 
            Monte Carlomethod. (default 0)

    Outputs:
    s     - frequency
    c     - coherence
    ph    - phase
    ci    - 95 coherence confidence level
    phi   - 95 phase confidence interval, bias corrected (add and subtract phi 
            from ph).
    """
    if NW<4:
        NW=8
    x = x - x.mean()
    y = y - y.mean()
    
    #define some parameters
    N = x.size 
    k = np.minimum(round(2*NW), N)
    k = int(np.maximum(k-1,1))
    s = np.linspace(0.0,1./1-1./(N*dt), N)
    pls = np.arange(2,(N+1)/2+1)
    pls = pls.astype(int)
    v = (2*NW-1) #approximate degrees of freedom
    
    if np.remainder(max(y.shape), 2) == 1:
        pls = pls[0:pls.shape[0] - 1]
        
    #Compute the discrete prolate spheroidal sequences, requires lbtfr.
    E, V = dpss(N, NW, k) 
    
    #Compute the windowed DFTs.
    fkx = np.fft.fft(E[0:k,:]*np.outer(np.ones(shape=(k)),x),N)
    fky = np.fft.fft(E[0:k,:]*np.outer(np.ones(shape=(k)),y),N)
    
    Pkx = abs(fkx) ** 2
    Pky = abs(fky) ** 2
    
    #Iteration to determine adaptive weights:    
    for i1 in range(1, 3):
        if i1 == 1:
            vari = sum(x*x)/N        
            Pk = Pkx        
        if i1 == 2:
            vari = sum(y*y)/N
            Pk = Pky        
        P = (Pk[0, :] + Pk[1, :]) / 2
        
        # initial spectrum estimate
        Ptemp = np.zeros(shape=(1, N), dtype='float64')
        P1 = np.zeros(shape=(1,N), dtype='float64')
        tol = (0.0005*vari) / N
        
        # usually within 'tol'erance in about three iterations, see equations from [2] (P&W pp 368-370).   
        a = vari* (1 - V)
        
        while np.sum(abs(P - P1) / N) > tol:        
            b=(np.outer(np.ones(shape=(1,k)),P))/(np.outer(V,P)+np.outer(a,np.ones(shape=(1,N)))) # weights        
            wk =(b**2)*(np.outer(V,np.ones(shape=(N,1)))) # new spectral estimate        
            P1 = sum(wk*Pk, axis=0)/sum(wk,axis=0)        
            Ptemp = P1; P1 = P; P = Ptemp # swap P and P1        
    
        if i1 == 1:
            fkx = np.sqrt(k)*np.sqrt(wk)*fkx / np.full((k,1), np.sum(np.sqrt(wk)))
            #fkx = sqrt(k)*sqrt(wk)*fkx / kron(ones((k,1)),np.sum(sqrt(wk))) 
            Fx = P #Power density spectral estimate of x        
            
        if i1 == 2:
            fky = np.sqrt(k)*np.sqrt(wk)*fky / np.full((k,1), np.sum(np.sqrt(wk)))
            
            #fky = sqrt(k)*sqrt(wk)*fky / kron(ones((k,1)),np.sum(sqrt(wk)))         
            Fy = P #Power density spectral estimate of y
    
    #As a check, the quantity sum(abs(fkx(pls,:))'.^2) is the same as Fx and
    #the spectral estimate from pmtmPH.
    
    #Compute the cross spectrum, phase delay and coherence
    Cxy = sum(fkx*fky.conjugate(),axis=0)
    ph = np.angle(Cxy)#*180/np.pi#deg=1)
    c = abs(Cxy) / sqrt(sum(abs(fkx)**2, axis=0) * sum(abs(fky)**2,axis=0))
    
    # if qbias ==1, correct coherence estimate for bias
    if qbias==1:
        c=cohbias(v,c)
    
    # confn is the number of iterations to use in estimating phase uncertainty via Monte Carlo analysis
    if confn >1:    
        cb=cohbias(v,c) # correct for bias
        
        #ciph = np.zeros((confn,len(pls)))
        phi = np.zeros((confn,len(pls)))
        
        for i in xrange(confn):
            # create semi-random timeseries using coherence estimate
            fx = np.fft.fft(randn(len(x))+1)
            fx = fx/sum(abs(fx))
            
            fy = np.fft.fft(randn(len(y))+1)
            fy = fy/sum(abs(fy))  
            
            ys = np.real(np.ifft(fy * sqrt(1 - cb ** 2))) 
            ys = ys + np.real(np.ifft(fx *cb))
            
            xs = np.real(np.ifft(fx))
            # use xs and ys as input for function to determine coherence and phase
            si, ciph,phi[i], phi_ignore,xx_ignor,yy_ignore,xy_ignore = cmtm(xs,ys,dt,NW)
        
        # sort and average the highest uncertianties
        pl = int(round(0.975*i+1))
        phi = np.sort(phi,axis=0)        
        phi = phi[((i+1)-pl):pl]
        phi = np.array([phi[pl-1,:],-phi[pl-confn,:]])
        phi = phi.mean(axis=0)#
        phi = np.convolve(phi, np.array([1,1,1])/3)
        phi = phi[1:-1]
        
    # Cut to one-sided functions
    Fx = Fx[pls]
    Fy = Fy[pls]
    Cxy = Cxy[pls]
    c = c[pls]
    s = s[pls]
    ph = ph[pls]
    
    try:
        phi = phi*1        
    except NameError:        
        phi = np.zeros_like(pls)        
    
    # Function returns: frequency (s), coherence (c), phase (ph), phase uncertainty (phi), 
    #                   power spectrum x (Fx), power spectrum y (Fy), cross spectrum (Cxy)
    return s,c,ph,phi,Fx,Fy,Cxy 


# In[ ]:

abs(-1)


# In[ ]:

end = 2000
t = np.arange(0,end)
pi_plu = np.full(len(t), pi)
pi_min = np.full(len(t), -pi)
rand1 = np.random.rand(end)
rand2 = np.random.rand(end)
x_1 = 1*cos(2*pi*90/360+2*pi/21*t)+4*cos(+2*pi*10/360-2*pi/10*t)+1.5*(-1+2*rand1)
y_1 = 1*cos(2*pi*60/360+2*pi/21*t)+4*cos(-2*pi*70/360-2*pi/10*t)+1.5*(-1+2*rand2)
plot(t,x_1)
plot(t,y_1)
xlim([0,100])
#plot(t,pi_plu)
#plot(t,pi_min)
sf,cf,phf,phif,fxf,fyf,cxyf = cmtm(x_1, y_1, qbias=1, confn=10)
pi_plu = np.full(len(sf), 2*pi)
pi_min = np.full(len(sf), -2*pi)
ylabel('amplitude')
xlabel('sample number')
title('First 100 samples of raw data series')
legend(['series 1', 'series 2'])
#savefig(r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM//sinus.png', dpi=200)


# In[ ]:

subplot(121)
plot(sf,10*np.log10(fxf))
plot(sf,10*np.log10(fyf))
#axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
#axvspan(0.045,0.050, color=GRAY, alpha=0.5)
p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)

xlim([0,0.013])
ylabel('power ($10log_{10}$)')
xlabel('frequency (Hertz)')
title('Power spectral density of precipitation and NDVI')
legend(['psd prec', 'psd NDVI'])
#savefig(r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM//psd.png', dpi=200)

subplot(122)
plot(sf,10*np.log10(cxyf))
p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)

#axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
#axvspan(0.045,0.050, color=GRAY, alpha=0.5)
xlim([0,0.013])
ylabel('power ($10log_{10}$)')
xlabel('frequency (Hertz)')
title('Cross spectral density of precipitation and NDVI')
legend(['csd prec and NDVI'])

tight_layout()
#savefig(r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//cpsd.png', dpi=200)


# In[ ]:

subplot(121)
plot(sf,cf)
#axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
#axvspan(0.045,0.050, color=GRAY, alpha=0.5)
p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)

xlim([0,0.013])
ylim([-0.1,1.1])
ylabel('coherence')
xlabel('frequency (Hertz)')
title('Coherence prec and NDVI')
legend(['coherence of prec and NDVI'], loc=4)

subplot(122)
plot(sf,phf, color=ORANGE)
#axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
#axvspan(0.045,0.050, color=GRAY, alpha=0.5)
p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)

xlim([0,0.013])
ylim([-pi-0.1,pi+0.1])
ylabel('phase (radian)')
xlabel('frequency (Hertz)')
title('Phase spectrum of prec and NDVI')
legend(['phase of prec and NDVI'], loc=3)

tight_layout()
#savefig(r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//cohphase.png', dpi=200)


# In[ ]:

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
host_subplot(111)
plot(sf,phif)
#axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
#axvspan(0.045,0.050, color=GRAY, alpha=0.5)
p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)

xlim([0,0.013])

ylabel('phase uncertainty (radian)')
xlabel('frequency (Hertz)')
title('Phase uncertainty of prec and NDVI via Monte Carlo analysis')
legend(['phase uncertainty of prec and NDVI'], loc=2)

tight_layout()
#savefig(r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//hase.png', dpi=200)


# In[ ]:

subplot(321)


subplot(322)
plot(sf,cf)
axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
axvspan(0.045,0.050, color=GRAY, alpha=0.5)
xlim([0,0.5])
ylim([-0.1,1.1])

subplot(323)
plot(sf,phf)
axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
axvspan(0.045,0.050, color=GRAY, alpha=0.5)
xlim([0,0.5])

subplot(324)
plot(sf,phif)
axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
axvspan(0.045,0.050, color=GRAY, alpha=0.5)
xlim([0,0.5])

subplot(325)
plot(sf,phf, color=BLACK)
#fill_between(sf,(phf+phif),(phf-phif), where=(((phf+phif)<(pi))),facecolor=ORANGE,edgecolor=WHITE,interpolate=True)
#fill_between(sf,(phf+phif),pi_plu, where=(((phf+phif)>(pi))),facecolor=ORANGE,edgecolor=WHITE,interpolate=True)
fill_between(sf,(phf+phif),(phf-phif),alpha=0.5)
fill_between(sf,((phf+phif)-pi_plu),pi_min, where=(((phf+phif)>pi_min)),alpha=0.5)
fill_between(sf,((phf-phif)+pi_plu),pi_plu, where=(((phf-phif)<-pi)),alpha=0.5)
axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
axvspan(0.045,0.050, color=GRAY, alpha=0.5)
xlim([0,0.013])
ylim([-pi,pi])

tight_layout()


# In[ ]:

plot(sf,phf, color=BLACK)
#fill_between(sf,(phf+phif),(phf-phif), where=(((phf+phif)<(pi))),facecolor=ORANGE,edgecolor=WHITE,interpolate=True)
#fill_between(sf,(phf+phif),pi_plu, where=(((phf+phif)>(pi))),facecolor=ORANGE,edgecolor=WHITE,interpolate=True)
fill_between(sf,(phf+phif),(phf-phif),alpha=0.5)
fill_between(sf,((phf+phif)-pi_plu),pi_min, where=(((phf+phif)>pi_min)),alpha=0.5)
fill_between(sf,((phf-phif)+pi_plu),pi_plu, where=(((phf-phif)<-pi)),alpha=0.5)
axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
axvspan(0.045,0.050, color=GRAY, alpha=0.5)
xlim([0,0.5])
ylim([-pi,pi])


# In[ ]:

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

rad2time = phf/(2*np.pi*sf)
mtcl2time = phif/(2*np.pi*sf)
neg_time= np.where(rad2time<0)
dur_cycl = (1/sf)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]

ax = host_subplot(111, axes_class=AA.Axes)

p1 = Rectangle((0, 0), 1, 1, fc=ORANGE, ec=WHITE, alpha=0.5)
p2, = ax.plot(sf, rad2time, color=BLACK, zorder=5, label='Phase')
p3, = ax.plot(sf, dur_cycl, color=GRAY, linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(sf, dur_cycl/2, color=GRAY, linestyle='--', zorder=5, label='Halve period')

ax.fill_between(sf,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4, alpha=0.5)
ax.fill_between(sf,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4, alpha=0.5)
ax.fill_between(sf,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4, alpha=0.5)
ax.fill_between(sf,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4, alpha=0.5)

ax.set_xlim([0,0.013])
ax.set_ylim([0,600])
#ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])

ax.text(0.125, 0.7,'Series 1 lags series 2', ha='center', va='center', rotation=0,transform=ax.transAxes, zorder=4, color='b')
ax.text(0.085, 0.175,'Series 1 leads series 2', ha='center', va='center', rotation=0,transform=ax.transAxes, zorder=4, color='b')

ax.set_xlabel('Frequency (Hertz)')
ax.set_ylabel('Phase-Delay (hours)')

#p5= axvspan(0.0975,0.1025, color=GRAY, alpha=0.5)
#axvspan(0.045,0.050, color=GRAY, alpha=0.5)

p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)


ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('The Dependence of Phase-Delay on Frequency and Time (+ Phase Uncertainty Estimates via Monte Carlo Analysis)')
ax2.set_xticks([(0.05),(0.1),(0.15),(0.2), (0.25)])#0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
ax2.set_xticklabels([str(1./0.05),str(1./0.1),str(round(1./0.15,1)),str(1./0.2),str(1./0.25)])
ax2.axis["right"].major_ticklabels.set_visible(False)

plt.legend([p2,p3,p4,p1,p5], ['phase-delay','single period','halve period','phase uncertainty estimates', 'frequencies of interest'])
plt.grid(True, zorder=0 )

plt.title('Time (period in hours)')
plt.tight_layout()
out_pth = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM//phase_delay_montecarlo.png'
#plt.savefig(out_pth, dpi=200)
print out_pth
plt.show()


# In[ ]:

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

rad2time = phf/(2*np.pi*sf)
mtcl2time = phif/(2*np.pi*sf)
neg_time= np.where(rad2time<0)
dur_cycl = (1/sf)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]

ax = host_subplot(111, axes_class=AA.Axes)

p1 = Rectangle((0, 0), 1, 1, fc=ORANGE, ec=WHITE)
p2, = ax.plot(sf, rad2time, color=BLACK, zorder=5, label='Phase')
p3, = ax.plot(sf, dur_cycl, color=GRAY, linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(sf, dur_cycl/2, color=GRAY, linestyle='--', zorder=5, label='Halve period')

ax.fill_between(sf,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)

ax.set_xlim([(1/(366*2)),0.013])
ax.set_ylim([0,370])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])

ax.text(0.125, 0.7,'NDVI leads Precipitation', ha='center', va='center', rotation=0,transform=ax.transAxes)
ax.text(0.085, 0.225,'NDVI lags Precipitation', ha='center', va='center', rotation=0,transform=ax.transAxes)

ax.set_xlabel('Frequency (Hertz)')
ax.set_ylabel('Phase-Delay (days)')

p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('The Dependence of Phase-Delay on Frequency and Time (+ Phase Uncertainty Estimates via Monte Carlo Analysis)')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])#0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)

plt.legend([p2,p3,p4,p1,p5], ['phase-delay','single period','halve period','phase uncertainty estimates', 'frequencies of interest'])
plt.grid(True, zorder=0 )

plt.title('Time (months)')
plt.tight_layout()
out_pth = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia//phase_delay_montecarlo.png'
#plt.savefig(out_pth, dpi=200)
print out_pth
plt.show()


# In[ ]:

mth = [0.5, 1, 2, 3, 4, 5]
prd = [15, 30, 60, 90, 120, 150]
pth1 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2_2009\XX_Day_Period\XX_DaySums_StdNormAnomaly'# removed 'NormTypeII'
pth2 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI_2009\DaySums_StdNormAnomalyRes'# removed 'NormTypeII'

#pixelcolumn = 23 # extract from rasterXSize (0:62) horizontal latitude - whatever
#pixelrow = 21 # extract from rasterYSize (0:36) vertical longitude 
#folderp1 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\PLOT_CHARTS_2009\COHERENCE2\p5//'
#folderp2 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\PLOT_CHARTS_2009\COHERENCE2\p4//'

#frequency bands of interest (12, 6, 4, 3 months)
annual_p = (1./(366+3))
annual_m = (1./(366-3))
binual_p = (1./((366.+3)/2))
binual_m = (1./((366.-3)/2))
trinual_p = (1./((366.+3)/3))
trinual_m = (1./((366.-3)/3))
quatro_p = (1./((366.+3)/4))
quatro_m = (1./((366.-3)/4))

ph = pth2+'//TRMM_CQ_2009001.tif'
ds = gdal.Open(ph)
base = ds.ReadAsArray()

for i in range(0,1):#6
    
    coher_array12 = np.zeros_like(base)
    coher_array06 = np.zeros_like(base)
    coher_array04 = np.zeros_like(base)
    coher_array03 = np.zeros_like(base)
    
    phase_array12 = np.zeros_like(base)
    phase_array06 = np.zeros_like(base)
    phase_array04 = np.zeros_like(base)
    phase_array03 = np.zeros_like(base)
    
    phunc_array12 = np.zeros_like(base)
    phunc_array06 = np.zeros_like(base)
    phunc_array04 = np.zeros_like(base)
    phunc_array03 = np.zeros_like(base)
    
    pwsdx_array12 = np.zeros_like(base)
    pwsdx_array06 = np.zeros_like(base)
    pwsdx_array04 = np.zeros_like(base)
    pwsdx_array03 = np.zeros_like(base)

    pwsdy_array12 = np.zeros_like(base)
    pwsdy_array06 = np.zeros_like(base)
    pwsdy_array04 = np.zeros_like(base)
    pwsdy_array03 = np.zeros_like(base)

    csdxy_array12 = np.zeros_like(base)
    csdxy_array06 = np.zeros_like(base)
    csdxy_array04 = np.zeros_like(base)
    csdxy_array03 = np.zeros_like(base)
    
    for j in range(36, 37): # 0,48 pixelcolumn
        for k in range(24,25): #0,35 pixelrow
        
            # Set path for cumulative period of precipitation
            # And corresponding period time
            cumper = mth[i]
            period = prd[i]
            paths1 = pth1.replace('XX', str(period))
            paths2 = pth2
            
    
            # Set pixelvalues
            # TODO: Implementation to read row by row or all-in-one
            pixelcolumn = j
            pixelrow = k
            
            # Get values for precipitation based on cumper 
            # And only update values for NDVI with a new pixel
            s_org2 = getvalues(paths2, pixelcolumn, pixelrow, wildcard='CQ')            
            if s_org2.mean()<>-99999.0:
                s_org1 = getvalues(paths1, pixelcolumn, pixelrow, wildcard='TRMM')                
            
                # Create dataframe from the two series and apply an outer join            
                df1 = pd.DataFrame(s_org1, columns=['prcp'])
                df2 = pd.DataFrame(s_org2, columns=['ndvi'])            
                df = pd.concat([df1, df2], join='outer', axis=1)
                
                # Before splitting the signals, first drop eventual NaN values
                df = df.dropna()            
                s1 = df['prcp']
                s2 = df['ndvi']
                
                sf,cf,phf,phif,fxf,fyf,cxyf = cmtm(np.resize(s1, (s1.size*50) ), np.resize(s2, (s2.size*50) ), qbias=1, confn=10)


# In[8]:

mth = [0.5, 1, 2, 3, 4, 5]
prd = [10, 60, 150]#[15, 30, 60, 90, 120, 150]
pth1 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2_2009\XX_Day_Period\XX_DaySums_StdNormAnomaly'# removed 'NormTypeII'
pth2 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI_2009\DaySums_StdNormAnomalyRes'# removed 'NormTypeII'

#pixelcolumn = 23 # extract from rasterXSize (0:62) horizontal latitude - whatever
#pixelrow = 21 # extract from rasterYSize (0:36) vertical longitude 
#folderp1 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\PLOT_CHARTS_2009\COHERENCE2\p5//'
#folderp2 = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\PLOT_CHARTS_2009\COHERENCE2\p4//'

#frequency bands of interest (12, 6, 4, 3 months)
annual_p = (1./(366+3))
annual_m = (1./(366-3))
binual_p = (1./((366.+3)/2))
binual_m = (1./((366.-3)/2))
trinual_p = (1./((366.+3)/3))
trinual_m = (1./((366.-3)/3))
quatro_p = (1./((366.+3)/4))
quatro_m = (1./((366.-3)/4))

ph = pth2+'//TRMM_CQ_2009001.tif'
ds = gdal.Open(ph)
base = ds.ReadAsArray()

for i in range(0,1):#6
    
    for j in range(25, 26): # 0,48 pixelcolumn
        for k in range(24,25): #0,35 pixelrow
        
            # Set path for cumulative period of precipitation
            # And corresponding period time
            cumper = mth[i]
            period = prd[i]
            paths1 = pth1.replace('XX', str(period))
            paths2 = pth2
            
    
            # Set pixelvalues
            # TODO: Implementation to read row by row or all-in-one
            pixelcolumn = j
            pixelrow = k
            
            # Get values for precipitation based on cumper 
            # And only update values for NDVI with a new pixel
            s_org2 = getvalues(paths2, pixelcolumn, pixelrow, wildcard='CQ')            
            if s_org2.mean()<>-99999.0:
                s_org1 = getvalues(paths1, pixelcolumn, pixelrow, wildcard='TRMM')                
            
                # Create dataframe from the two series and apply an outer join            
                df1 = pd.DataFrame(s_org1, columns=['prcp'])
                df2 = pd.DataFrame(s_org2, columns=['ndvi'])            
                df = pd.concat([df1, df2], join='outer', axis=1)
                
                # Before splitting the signals, first drop eventual NaN values
                df = df.dropna()            
                s1 = df['prcp']
                s2 = df['ndvi']


# In[190]:

pixelcolumn = 497#597
pixelrow = 528#428
s_org2 = getvalues(paths2, pixelcolumn, pixelrow, wildcard='CQ')
s2 = s_org2.copy()


# In[191]:

# NDVI
s2.fillna(method='ffill', inplace=True)
s2.plot()


# In[86]:

# NDVI pxcol = 597 , pxrow = 428
s2.fillna(method='ffill', inplace=True)
s2.plot()


# In[4]:

s1_150 = s1.copy()


# In[5]:

# 150-day composite precipitation
s1_150.plot()


# In[7]:

s1_60 = s1.copy()


# In[9]:

s1_10 = s1.copy()


# In[192]:

y1_10 = HANTS(s1_10.shape[-1], s1_10.as_matrix()[None], 1, outliers_to_reject='Lo', 
               low=-2000, high = 10000, fit_error_tolerance=500, base=1)
ts1_10 = pd.Series(y1_10[0], index = s1_10.index)

y1_60 = HANTS(s1_60.shape[-1], s1_60.as_matrix()[None], 1, outliers_to_reject='Lo', 
               low=-2000, high = 10000, fit_error_tolerance=500, base=1)
ts1_60 = pd.Series(y1_60[0], index = s1_60.index)

y1_150 = HANTS(s1_150.shape[-1], s1_150.as_matrix()[None], 1, outliers_to_reject='Lo', 
               low=-2000, high = 10000, fit_error_tolerance=500, base=1)
ts1_150 = pd.Series(y1_150[0], index = s1_150.index)

y2 = HANTS(s2.shape[-1], s2.as_matrix()[None], 1, outliers_to_reject='Lo', 
               low=-2000, high = 10000, fit_error_tolerance=500, base=1)
ts2 = pd.Series(y2[0], index = s2.index)


# In[193]:

vspan_start = 0.00255
vspan_end = 0.00285
xlim_start = 0.000
xlim_end = 0.012#0.012


# In[204]:

plt.figure(figsize=(12,4))
plt.subplot(121)
s1_150.plot(label='150-day P composite')
s1_60.plot(label='60-day P composite',color='k')
s1_10.plot(label='10-day P composite', color='m')
s2.plot(label='NDVI', linewidth=2)
from scipy import signal
b, a = signal.butter(1, 0.01)
y = signal.filtfilt(b,a,s1_10, padlen=150)
plt.plot(y)
plt.ylim([-0.65,0.4])
plt.axhline(color='k', ls='--')
leg = plt.legend(loc=3)
leg.get_frame().set_linewidth(0.0)
plt.title('Standardized anomaly P and NDVI')
plt.ylabel('Anomaly difference')
plt.tight_layout()
plt.grid(axis='y')
out_pth = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//prec_ndvi_all.png'
#savefig(out_pth, dpi=200)
print out_pth

plt.subplot(122)
ts1_150.plot(label='150-day P annual')
ts1_60.plot(label='60-day P annual',color='k')
ts1_10.plot(label='10-day P annual', color='m')
ts2.plot(label='NDVI', linewidth=2)
from scipy import signal
b, a = signal.butter(1, 0.01)
y = signal.filtfilt(b,a,s1_10, padlen=150)
plt.plot(y)
plt.ylim([-0.65,0.4])
plt.axhline(color='k', ls='--')
leg = plt.legend(loc=3)
leg.get_frame().set_linewidth(0.0)
plt.title('Annual component P and NDVI')
plt.ylabel('Anomaly difference')
plt.tight_layout()
plt.grid(axis='y')
out_pth = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//prec_ndvi_raw_annuall_1.png'
plt.savefig(out_pth, dpi=200)
print out_pth
plt.show()


# In[195]:

tile_no = 50
x_10, y_10, t = pad_extend(s1_10, s2, tile_no=tile_no, extend = 0)
f_10, fkx, fky, cxy_10, ph_10, coh_10 = mtem(x_10,y_10)
phif_10 = mtem_unct(x_10,y_10,coh_10, mc_no=5)

tile_no = 50
x_60, y_60, t = pad_extend(s1_60, s2, tile_no=tile_no, extend = 0)
f_60, fkx, fky, cxy_60, ph_60, coh_60 = mtem(x_60,y_60)
phif_60 = mtem_unct(x_60,y_60,coh_60, mc_no=5)

tile_no = 50
x_150, y_150, t = pad_extend(s1_150, s2, tile_no=tile_no, extend = 0)
f_150, fkx, fky, cxy_150, ph_150, coh_150 = mtem(x_150,y_150)
phif_150 = mtem_unct(x_150,y_150,coh_150, mc_no=5)


# In[196]:

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.grid(axis='y')
plt.plot(f_10,10*np.log10(cxy_10), 'k')
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(-10,20)

plt.ylabel('cross-power (10log10)')
plt.xlabel('frequency (Hz)')
plt.title('cross-power: 10-day')

plt.subplot(132)
plt.grid(axis='y')
plt.plot(f_60,10*np.log10(cxy_60), 'k')
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(-10,20)

plt.ylabel('cross-power (10log10)')
plt.xlabel('frequency (Hz)')
plt.title('cross-power: 60-day')

plt.subplot(133)
plt.grid(axis='y')
plt.plot(f_150,10*np.log10(cxy_150), 'k')
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(-10,20)

plt.ylabel('cross-power (10log10)')
plt.xlabel('frequency (Hz)')
plt.title('cross-power: 150-day')

plt.tight_layout()

out_pth = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//prec_ndvi_cross-power.png'
#plt.savefig(out_pth, dpi=200)


# In[200]:

plt.figure(figsize=(12,3))
## PLOT 9
rad2time = ph_10/(2*np.pi*f_10)
mtcl2time = phif_10/(2*np.pi*f_10)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f_10)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]

#frequency index
f_lb = vspan_start 
f_ub = vspan_end
f_ix = np.where((f_10 > f_lb) * (f_10 < f_ub))[0]
p_r2t = np.mean(rad2time[f_ix], -1)
p_ph = np.mean(ph_10[f_ix], -1)
print ('phase in radian is', round(p_ph,2))
print ('which correspond to', round(p_r2t,2), 'day')


ax = host_subplot(131, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f_10, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f_10, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f_10, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f_10,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_10,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_10,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_10,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.2, zorder=-1)
plt.axvspan(0.0054,0.0056, color='gray', alpha=0.2)
plt.axvspan(0.0081,0.0083, color='gray', alpha=0.2)
plt.axvspan(0.0108,0.0110, color='gray', alpha=0.2)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum x/y + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['single period','half period'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

plt.gcf().tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME//nodetrending_2007_Wet_nohants.png', dpi=400)

# frequency index
#f_lb = 0.0054 # vspan_start 
#f_ub = 0.0056 # vspan_end
#f_ix = np.where((f > f_lb) * (f < f_ub))[0]
#p_r2t = np.mean(rad2time[f_ix], -1)
#p_ph = np.mean(ph[f_ix], -1)
#print ('phase in radian is', round(p_ph,2))
#print ('which correspond to', round(p_r2t,2), 'day')
#plt.savefig(r'C:\Users\lenovo\Documents\HOME\2015-02-05_pics4progress//nodetrending_2006_Wet_nohants.png', dpi=400)

## PLOT 9
rad2time = ph_60/(2*np.pi*f_60)
mtcl2time = phif_60/(2*np.pi*f_60)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f_60)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]

#frequency index
f_lb = vspan_start 
f_ub = vspan_end
f_ix = np.where((f_10 > f_lb) * (f_10 < f_ub))[0]
p_r2t = np.mean(rad2time[f_ix], -1)
p_ph = np.mean(ph_10[f_ix], -1)
print ('phase in radian is', round(p_ph,2))
print ('which correspond to', round(p_r2t,2), 'day')


ax = host_subplot(132, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f_60, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f_60, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f_60, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f_60,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_60,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_60,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_60,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.2, zorder=-1)
plt.axvspan(0.0054,0.0056, color='gray', alpha=0.2)
plt.axvspan(0.0081,0.0083, color='gray', alpha=0.2)
plt.axvspan(0.0108,0.0110, color='gray', alpha=0.2)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum x/y + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['single period','half period'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

plt.gcf().tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME//nodetrending_2007_Wet_nohants.png', dpi=400)

# frequency index
#f_lb = 0.0054 # vspan_start 
#f_ub = 0.0056 # vspan_end
#f_ix = np.where((f > f_lb) * (f < f_ub))[0]
#p_r2t = np.mean(rad2time[f_ix], -1)
#p_ph = np.mean(ph[f_ix], -1)
#print ('phase in radian is', round(p_ph,2))
#print ('which correspond to', round(p_r2t,2), 'day')
#plt.savefig(r'C:\Users\lenovo\Documents\HOME\2015-02-05_pics4progress//nodetrending_2006_Wet_nohants.png', dpi=400)

## PLOT 9
rad2time = ph_150/(2*np.pi*f_150)
mtcl2time = phif_150/(2*np.pi*f_150)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f_150)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]

#frequency index
f_lb = vspan_start 
f_ub = vspan_end
f_ix = np.where((f_10 > f_lb) * (f_10 < f_ub))[0]
p_r2t = np.mean(rad2time[f_ix], -1)
p_ph = np.mean(ph_10[f_ix], -1)
print ('phase in radian is', round(p_ph,2))
print ('which correspond to', round(p_r2t,2), 'day')


ax = host_subplot(133, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f_150, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f_150, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f_150, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f_150,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_150,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_150,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_150,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.2, zorder=-1)
plt.axvspan(0.0054,0.0056, color='gray', alpha=0.2)
plt.axvspan(0.0081,0.0083, color='gray', alpha=0.2)
plt.axvspan(0.0108,0.0110, color='gray', alpha=0.2)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum x/y + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['single period','half period'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

plt.tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME//nodetrending_2007_Wet_nohants.png', dpi=400)

# frequency index
#f_lb = 0.0054 # vspan_start 
#f_ub = 0.0056 # vspan_end
#f_ix = np.where((f > f_lb) * (f < f_ub))[0]
#p_r2t = np.mean(rad2time[f_ix], -1)
#p_ph = np.mean(ph[f_ix], -1)
#print ('phase in radian is', round(p_ph,2))
#print ('which correspond to', round(p_r2t,2), 'day')
#plt.savefig(r'C:\Users\lenovo\Documents\HOME\2015-02-05_pics4progress//nodetrending_2006_Wet_nohants.png', dpi=400)



plt.show()


# In[199]:

#frequency index
f_lb = vspan_start 
f_ub = vspan_end
f_ix = np.where((f_10 > f_lb) * (f_10 < f_ub))[0]
p_r2t = np.mean(rad2time[f_ix], -1)
p_ph = np.mean(ph_10[f_ix], -1)
print ('phase in radian is', round(p_ph,2))
print ('which correspond to', round(p_r2t,2), 'day')


# In[201]:

plt.figure(figsize=(12,6))
plt.subplot(231)
plt.grid(axis='y')
plt.plot(f_10,10*np.log10(cxy_10), 'k')
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(-10,20)

plt.ylabel('cross-power (10log10)')
plt.xlabel('frequency (Hz)')
plt.title('cross-power: 10-day')

plt.subplot(232)
plt.grid(axis='y')
plt.plot(f_60,10*np.log10(cxy_60), 'k')
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(-10,20)

plt.ylabel('cross-power (10log10)')
plt.xlabel('frequency (Hz)')
plt.title('cross-power: 60-day')

plt.subplot(233)
plt.grid(axis='y')
plt.plot(f_150,10*np.log10(cxy_150), 'k')
plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
plt.xlim(xlim_start,xlim_end)
plt.ylim(-10,20)

plt.ylabel('cross-power (10log10)')
plt.xlabel('frequency (Hz)')
plt.title('cross-power: 150-day')

plt.tight_layout()

out_pth = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//prec_ndvi_cross-power.png'
#plt.savefig(out_pth, dpi=200)

#plt.figure(figsize=(12,3))
## PLOT 9
rad2time = ph_10/(2*np.pi*f_10)
mtcl2time = phif_10/(2*np.pi*f_10)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f_10)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]


ax = host_subplot(234, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f_10, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f_10, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f_10, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f_10,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_10,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_10,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_10,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)


p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
#plt.axvspan(0.0054,0.0056, color='gray', alpha=0.2)
#plt.axvspan(0.0081,0.0083, color='gray', alpha=0.2)
#plt.axvspan(0.0108,0.0110, color='gray', alpha=0.2)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum 10-day P / NDVI + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['sync','out-of-sync'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

#plt.gcf().tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME//nodetrending_2007_Wet_nohants.png', dpi=400)

# frequency index
#f_lb = 0.0054 # vspan_start 
#f_ub = 0.0056 # vspan_end
#f_ix = np.where((f > f_lb) * (f < f_ub))[0]
#p_r2t = np.mean(rad2time[f_ix], -1)
#p_ph = np.mean(ph[f_ix], -1)
#print ('phase in radian is', round(p_ph,2))
#print ('which correspond to', round(p_r2t,2), 'day')
#plt.savefig(r'C:\Users\lenovo\Documents\HOME\2015-02-05_pics4progress//nodetrending_2006_Wet_nohants.png', dpi=400)

## PLOT 9
rad2time = ph_60/(2*np.pi*f_60)
mtcl2time = phif_60/(2*np.pi*f_60)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f_60)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]


ax = host_subplot(235, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f_60, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f_60, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f_60, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f_60,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_60,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_60,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_60,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
#plt.axvspan(0.0054,0.0056, color='gray', alpha=0.2)
#plt.axvspan(0.0081,0.0083, color='gray', alpha=0.2)
#plt.axvspan(0.0108,0.0110, color='gray', alpha=0.2)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum 60-day P / NDVI + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['sync','out-of-sync'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

#plt.gcf().tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME//nodetrending_2007_Wet_nohants.png', dpi=400)

# frequency index
#f_lb = 0.0054 # vspan_start 
#f_ub = 0.0056 # vspan_end
#f_ix = np.where((f > f_lb) * (f < f_ub))[0]
#p_r2t = np.mean(rad2time[f_ix], -1)
#p_ph = np.mean(ph[f_ix], -1)
#print ('phase in radian is', round(p_ph,2))
#print ('which correspond to', round(p_r2t,2), 'day')
#plt.savefig(r'C:\Users\lenovo\Documents\HOME\2015-02-05_pics4progress//nodetrending_2006_Wet_nohants.png', dpi=400)

## PLOT 9
rad2time = ph_150/(2*np.pi*f_150)
mtcl2time = phif_150/(2*np.pi*f_150)
neg_time= np.where(rad2time<0)
dur_cycl = (1/f_150)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]


ax = host_subplot(236, axes_class=AA.Axes)

p1 = plt.Rectangle((0, 0), 1, 1, fc='c', ec='c')
p2, = ax.plot(f_150, rad2time, color='m', zorder=5, label='Phase')
p3, = ax.plot(f_150, dur_cycl, color='gray', linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(f_150, dur_cycl/2, color='gray', linestyle='--', zorder=5, label='Halve period')

ax.fill_between(f_150,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_150,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_150,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)
ax.fill_between(f_150,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor='c' ,edgecolor='c', lw=0.0 ,interpolate=True, zorder=4)

ax.set_xlim([xlim_start,xlim_end])
ax.set_ylim([0,365])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])

ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('phase (day)')
ax.set_title('t (month)', loc='left', fontsize=10)

p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.5)
#plt.axvspan(0.0054,0.0056, color='gray', alpha=0.2)
#plt.axvspan(0.0081,0.0083, color='gray', alpha=0.2)
#plt.axvspan(0.0108,0.0110, color='gray', alpha=0.2)
#axvspan(0.099,0.101, color='gray', alpha=0.1, zorder=0)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('phase spectrum 150-day P / NDVI + uncertainty')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.xaxis.label.set_size(2)


lg = plt.legend([p3,p4], ['sync','out-of-sync'], ncol=1)
lg.get_frame().set_ec('lightgray')
lg.get_frame().set_lw(0.5)
plt.grid(axis='y', zorder=0 )

plt.tight_layout()
#plt.savefig(r'C:\Users\lenovo\Documents\HOME//nodetrending_2007_Wet_nohants.png', dpi=400)

# frequency index
#f_lb = 0.0054 # vspan_start 
#f_ub = 0.0056 # vspan_end
#f_ix = np.where((f > f_lb) * (f < f_ub))[0]
#p_r2t = np.mean(rad2time[f_ix], -1)
#p_ph = np.mean(ph[f_ix], -1)
#print ('phase in radian is', round(p_ph,2))
#print ('which correspond to', round(p_r2t,2), 'day')
plt.savefig(r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//cross_power_phase_new.png', dpi=200)



plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[31]:

get_ipython().magic(u'matplotlib inline')
from __future__ import division
from osgeo import gdal
import numpy as np
import scipy.signal
import os
import matplotlib.pyplot as plt
import nitime.algorithms as tsa
import random
import os
from osgeo import gdal
import scipy.signal
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


# In[32]:

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
    print ('x size', x.shape)
    print ('y size', y.shape)
    
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
    print ('iteration no is', mc_no)
    
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

def pad_extend(m1, m2, tile_no=100, extend=0):
    """
    Input 'm1':
    Signal 1
    Input 'm2':
    Signal 2
    Input 'extend':
    0 = repeat signal
    1 = padd with zeroes left
    2 = padd with zeroes right
    3 = padd with zeroes on both sides
    Input 'tile_no':
    number of times to pad or extend the signal
    
    Output 
    x,y,t = repeated/padded signals + evenly spaced interval of the range
    """
    
    
    if extend == 0:
        x = np.tile(m1, tile_no+1)
        y = np.tile(m2, tile_no+1)
    elif extend == 1:
        x = np.lib.pad(m1, (len(m1)*(tile_no),0), 'constant', constant_values=0)
        y = np.lib.pad(m2, (len(m2)*(tile_no),0), 'constant', constant_values=0)    
    
    elif extend == 2:
        x = np.lib.pad(m1, (0,len(m1)*(tile_no)), 'constant', constant_values=0)
        y = np.lib.pad(m2, (0,len(m1)*(tile_no)), 'constant', constant_values=0)    
    
    elif extend == 3:
        x = np.lib.pad(m1, (len(m1)*tile_no/2,len(m1)*(tile_no/2)), 'constant', constant_values=0)
        y = np.lib.pad(m2, (len(m2)*tile_no/2,len(m1)*(tile_no/2)), 'constant', constant_values=0)    
    
    if len(x) % 2 != 0:
        x = x[:-1]
    
    if len(y) % 2 != 0:
        y = y[:-1]
        
    t = np.arange(x.shape[0])
    return x,y,t


def plotCrossSpectral(f,ph,coh,phif):
    plt.figure(figsize=(14,4.5))
    
    ## PLOT 5
    plt.subplot(121)
    
    plt.grid(axis='y')
    plt.plot(f,coh, 'y')
    
    plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.2)
    plt.axvspan(0.0054,0.0056, color='gray', alpha=0.2)
    plt.axvspan(0.0081,0.0083, color='gray', alpha=0.2)
    plt.axvspan(0.0108,0.0110, color='gray', alpha=0.2)
    
    plt.xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))],[str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
    plt.xlim(xlim_start,xlim_end)
    plt.ylim(0,1.01)
    
    plt.ylabel('coherence')
    plt.xlabel('frequency (Hz)')
    plt.title('coherence x/y with repeated ts: '+str(tile_no))
    
    ## PLOT 9
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
    ax.set_ylim([0,365])
    ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])
    ax.set_xticklabels([str(0.0027),str(0.0055),str(0.0082),str(0.0109)])
    
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('phase (day)')
    ax.set_title('t (month)', loc='left', fontsize=10)
    
    p5= plt.axvspan(vspan_start,vspan_end, color='gray', alpha=0.2, zorder=-1)
    plt.axvspan(0.0054,0.0056, color='gray', alpha=0.2)
    plt.axvspan(0.0081,0.0083, color='gray', alpha=0.2)
    plt.axvspan(0.0108,0.0110, color='gray', alpha=0.2)
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
    #plt.savefig(r'C:\Users\lenovo\Documents\HOME//nodetrending_2007_Wet_nohants.png', dpi=400)
    
    # frequency index
    #f_lb = 0.0054 # vspan_start 
    #f_ub = 0.0056 # vspan_end
    #f_ix = np.where((f > f_lb) * (f < f_ub))[0]
    #p_r2t = np.mean(rad2time[f_ix], -1)
    #p_ph = np.mean(ph[f_ix], -1)
    #print ('phase in radian is', round(p_ph,2))
    #print ('which correspond to', round(p_r2t,2), 'day')
    #plt.savefig(r'C:\Users\lenovo\Documents\HOME\2015-02-05_pics4progress//nodetrending_2006_Wet_nohants.png', dpi=400)
    plt.show()
    plt.close()


# In[60]:

# --------*-*-*-*-*-*--------  HANTS code  --------*-*-*-*-*-*--------  HANTS code  --------*-*-*-*-*-*-----

# Computing diagonal for each row of a 2d array. See: http://stackoverflow.com/q/27214027/2459096
def makediag3d(M):
    b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
    b[:, ::M.shape[1] + 1] = M
    return b.reshape(M.shape[0], M.shape[1], M.shape[1]) 

def get_starter_matrix(base_period_len, sample_count, frequencies_considered_count):
    nr = min(2 * frequencies_considered_count + 1,
                  sample_count)  # number of 2*+1 frequencies, or number of input images
    mat = np.zeros(shape=(nr, sample_count))
    mat[0, :] = 1
    ang = 2 * np.pi * np.arange(base_period_len) / base_period_len
    cs = np.cos(ang)
    sn = np.sin(ang)
    # create some standard sinus and cosinus functions and put in matrix
    i = np.arange(1, frequencies_considered_count + 1)
    ts = np.arange(sample_count)
    for column in xrange(sample_count):
        index = np.mod(i * ts[column], base_period_len)
        # index looks like 000, 123, 246, etc, until it wraps around (for len(i)==3)
        mat[2 * i - 1, column] = cs.take(index)
        mat[2 * i, column] = sn.take(index)

    return mat        

def HANTS(sample_count, inputs,
          frequencies_considered_count=3,
          outliers_to_reject='Hi',
          low=0., high=255,
          fit_error_tolerance=5,
          delta=0.1, 
          base=2):
    """
    Function to apply the Harmonic analysis of time series applied to arrays

    sample_count    = nr. of images (total number of actual samples of the time series)
    base_period_len    = length of the base period, measured in virtual samples
            (days, dekads, months, etc.)
    frequencies_considered_count    = number of frequencies to be considered above the zero frequency
    inputs     = array of input sample values (e.g. NDVI values)
    ts    = array of size sample_count of time sample indicators
            (indicates virtual sample number relative to the base period);
            numbers in array ts maybe greater than base_period_len
            If no aux file is used (no time samples), we assume ts(i)= i,
            where i=1, ..., sample_count
    outliers_to_reject  = 2-character string indicating rejection of high or low outliers
            select from 'Hi', 'Lo' or 'None'
    low   = valid range minimum
    high  = valid range maximum (values outside the valid range are rejeced
            right away)
    fit_error_tolerance   = fit error tolerance (points deviating more than fit_error_tolerance from curve
            fit are rejected)
    dod   = degree of overdeterminedness (iteration stops if number of
            points reaches the minimum required for curve fitting, plus
            dod). This is a safety measure
    delta = small positive number (e.g. 0.1) to suppress high amplitudes
    base  = define the base period, default is 1, meaning it is equal to size as the input 
            values, if 2 it will double this period. So for annual series, the base period becomes
            two years
    """

    # define some parameters
    if base == 1:
        base_period_len = sample_count  #
    if base == 2:
        base_period_len = sample_count*2  #

    # check which setting to set for outlier filtering
    if outliers_to_reject == 'Hi':
        sHiLo = -1
    elif outliers_to_reject == 'Lo':
        sHiLo = 1
    else:
        sHiLo = 0

    nr = min(2 * frequencies_considered_count + 1,
             sample_count)  # number of 2*+1 frequencies, or number of input images

    # create empty arrays to fill
    outputs = np.zeros(shape=(inputs.shape[0], sample_count))

    mat = get_starter_matrix(base_period_len, sample_count, frequencies_considered_count)

    # repeat the mat array over the number of arrays in inputs
    # and create arrays with ones with shape inputs where high and low values are set to 0
    mat = np.tile(mat[None].T, (1, inputs.shape[0])).T
    p = np.ones_like(inputs)
    p[(low >= inputs) | (inputs > high)] = 0
    nout = np.sum(p == 0, axis=-1)  # count the outliers for each timeseries

    # prepare for while loop
    ready = np.zeros((inputs.shape[0]), dtype=bool)  # all timeseries set to false

    dod = 1  # (2*frequencies_considered_count-1)  # Um, no it isn't :/
    noutmax = sample_count - nr - dod
    # prepare to add delta to suppress high amplitudes but not for [0,0]
    Adelta = np.tile(np.diag(np.ones(nr))[None].T, (1, inputs.shape[0])).T * delta
    Adelta[:, 0, 0] -= delta
    
    for _ in xrange(sample_count):
        if ready.all():
            break
        # print '--------*-*-*-*',it.value, '*-*-*-*--------'
        # multiply outliers with timeseries
        za = np.einsum('ijk,ik->ij', mat, p * inputs)

        # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
        diag = makediag3d(p)
        A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
        # add delta to suppress high amplitudes but not for [0,0]
        A += Adelta
        #A[:, 0, 0] = A[:, 0, 0] - delta

        # solve linear matrix equation and define reconstructed timeseries
        zr = np.linalg.solve(A, za)
        outputs = np.einsum('ijk,kj->ki', mat.T, zr)

        # calculate error and sort err by index
        err = p * (sHiLo * (outputs - inputs))
        rankVec = np.argsort(err, axis=1, )

        # select maximum error and compute new ready status
        maxerr = np.max(err, axis=-1)
        #maxerr = np.diag(err.take(rankVec[:, sample_count - 1], axis=-1))
        ready = (maxerr <= fit_error_tolerance) | (nout == noutmax)

        # if ready is still false
        if not ready.all():
            j = rankVec.take(sample_count - 1, axis=-1)

            p.T[j.T, np.indices(j.shape)] = p.T[j.T, np.indices(j.shape)] * ready.astype(
                int)  #*check
            nout += 1

    return outputs


# Compute semi-random time series array with numb standing for number of timeseries
def array_in(numb):
    y = np.array([5.0, 2.0, 10.0, 12.0, 18.0, 23.0, 27.0, 40.0, 60.0, 70.0, 90.0, 160.0, 190.0,
                  210.0, 104.0, 90.0, 170.0, 50.0, 120.0, 60.0, 40.0, 30.0, 28.0, 24.0, 15.0,
                  10.0])
    y = np.tile(y[None].T, (1, numb)).T
    kl = (np.random.randint(2, size=(numb, 26)) *
          np.random.randint(2, size=(numb, 26)) + 1)
    kl[kl == 2] = 0
    y = y * kl
    return y


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

from scipy import signal
b, a = signal.butter(1, 0.01)
y = signal.filtfilt(b,a,s1, padlen=150)
plot(y)


# In[ ]:

x = np.resize(s1, (s1.size*100) )
y = np.resize(s2, (s2.size*100) )
dt = 1
NW = 8

"""
Multi-taper method coherence using adaptive weighting and correcting for the bias inherent to coherence estimates.  
The 95 coherence confidence level is computed by cohconf.m.  In addition, a built-in Monte Carlo estimation procedure 
is available to estimate phase 95 confidence limits. 

Inputs:
x     - Input data vector 1.  
y     - Input data vector 2.  
dt    - Sampling interval (default 1) 
NW    - Number of windows to use (default 8) 
qbias - Correct coherence estimate for bias (yes, 1)  (no, 0, default).
confn - Number of iterations to use in estimating phase uncertainty using a 
        Monte Carlomethod. (default 0)

Outputs:
s     - frequency
c     - coherence
ph    - phase
ci    - 95 coherence confidence level
phi   - 95 phase confidence interval, bias corrected (add and subtract phi 
        from ph).
"""
if NW<4:
    NW=8
x = x - x.mean()
y = y - y.mean()

#define some parameters
N = x.size 
k = np.minimum(round(2*NW), N)
k = int(np.maximum(k-1,1))
s = linspace(0.0,1./1-1./(N*dt), N)
pls = np.arange(2,(N+1)/2+1)
pls = pls.astype(int)
v = (2*NW-1) #approximate degrees of freedom

if remainder(max(y.shape), 2) == 1:
    pls = pls[0:pls.shape[0] - 1]
    
#Compute the discrete prolate spheroidal sequences, requires lbtfr.
E, V = dpss(N, NW, k) 

#Compute the windowed DFTs.
fkx = fft.fft(E[0:k,:]*np.outer(np.ones(shape=(k)),x),N)
fky = fft.fft(E[0:k,:]*np.outer(np.ones(shape=(k)),y),N)

Pkx = abs(fkx) ** 2
Pky = abs(fky) ** 2

#Iteration to determine adaptive weights:    
for i1 in range(1, 3):
    if i1 == 1:
        vari = sum(x*x)/N        
        Pk = Pkx        
    if i1 == 2:
        vari = sum(y*y)/N
        Pk = Pky        
    P = (Pk[0, :] + Pk[1, :]) / 2
    
    # initial spectrum estimate
    Ptemp = np.zeros(shape=(1, N), dtype='float64')
    P1 = np.zeros(shape=(1,N), dtype='float64')
    tol = (0.0005*vari) / N
    
    # usually within 'tol'erance in about three iterations, see equations from [2] (P&W pp 368-370).   
    a = vari* (1 - V)
    
    while np.sum(abs(P - P1) / N) > tol:        
        b=(np.outer(np.ones(shape=(1,k)),P))/(np.outer(V,P)+np.outer(a,np.ones(shape=(1,N)))) # weights        
        wk =(b**2)*(np.outer(V,np.ones(shape=(N,1)))) # new spectral estimate        
        P1 = sum(wk*Pk, axis=0)/sum(wk,axis=0)        
        Ptemp = P1; P1 = P; P = Ptemp # swap P and P1        

    if i1 == 1:
        fkx = sqrt(k)*sqrt(wk)*fkx / kron(ones((k,1)),np.sum(sqrt(wk))) 
        Fx = P #Power density spectral estimate of x        
        
    if i1 == 2:
        fky = sqrt(k)*sqrt(wk)*fky / kron(ones((k,1)),np.sum(sqrt(wk)))         
        Fy = P #Power density spectral estimate of y

#As a check, the quantity sum(abs(fkx(pls,:))'.^2) is the same as Fx and
#the spectral estimate from pmtmPH.

#Compute the cross spectrum, phase delay and coherence
Cxy = sum(fkx*fky.conjugate(),axis=0)
ph = angle(Cxy)#*180/np.pi#deg=1)
c = abs(Cxy) / sqrt(sum(abs(fkx)**2, axis=0) * sum(abs(fky)**2,axis=0))
    
# Cut to one-sided functions
#Fx = Fx[pls]
#Fy = Fy[pls]
#Cxy = Cxy[pls]
#c = c[pls]
#s = s[pls]
#ph = ph[pls]


# In[ ]:

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

rad2time = phf/(2*np.pi*sf)
mtcl2time = phif/(2*np.pi*sf)
neg_time= np.where(rad2time<0)
dur_cycl = (1/sf)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]

ax = host_subplot(111, axes_class=AA.Axes)

p1 = Rectangle((0, 0), 1, 1, fc=ORANGE, ec=WHITE)
p2, = ax.plot(sf, rad2time, color=BLACK, zorder=5, label='Phase')
p3, = ax.plot(sf, dur_cycl, color=GRAY, linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(sf, dur_cycl/2, color=GRAY, linestyle='--', zorder=5, label='Halve period')

ax.fill_between(sf,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)

ax.set_xlim([(1/(366*2)),0.013])
ax.set_ylim([0,370])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])

#ax.text(0.125, 0.7,'NDVI leads Precipitation', ha='center', va='center', rotation=0,transform=ax.transAxes)
#ax.text(0.085, 0.225,'NDVI lags Precipitation', ha='center', va='center', rotation=0,transform=ax.transAxes)

ax.set_xlabel('Frequency (Hertz)')
ax.set_ylabel('Phase-Delay (days)')

p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('The Dependence of Phase-Delay on Frequency and Time (+ Phase Uncertainty Estimates via Monte Carlo Analysis)')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])#0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)

plt.legend([p2,p3,p4,p1,p5], ['phase-delay','single period','halve period','phase uncertainty estimates', 'frequencies of interest'])
plt.grid(True, zorder=0 )

plt.title('Time (months)')
plt.tight_layout()
out_pth = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\png//phase_delay_montecarlo.png'
plt.savefig(out_pth, dpi=200)
print out_pth
plt.show()


# # THIS CODE BELOW IS DAR GOOD, PLOT FAKKING NICE PLOT!

# In[ ]:

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

rad2time = ph/(2*np.pi*s)
mtcl2time = phif/(2*np.pi*sf)
neg_time= np.where(rad2time<0)
dur_cycl = (1/s)
rad2time[neg_time] = rad2time[neg_time]+dur_cycl[neg_time]

ax = host_subplot(111, axes_class=AA.Axes)

p1 = Rectangle((0, 0), 1, 1, fc=ORANGE, ec=WHITE)
p2, = ax.plot(s, rad2time, color=BLACK, zorder=5, label='Phase')
p3, = ax.plot(s, dur_cycl, color=GRAY, linestyle='-.', zorder=5, label='Period')
p4, = ax.plot(s, dur_cycl/2, color=GRAY, linestyle='--', zorder=5, label='Halve period')

ax.fill_between(sf,(rad2time+mtcl2time),(rad2time-mtcl2time), where=(((rad2time+mtcl2time)<dur_cycl)), 
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,(rad2time-mtcl2time),dur_cycl, where=(((rad2time+mtcl2time)>dur_cycl)),
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,(rad2time+mtcl2time)-dur_cycl, where=(((rad2time+mtcl2time)>0)),
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)
ax.fill_between(sf,((rad2time-mtcl2time)+dur_cycl),dur_cycl, where=((rad2time-mtcl2time)<0), 
             facecolor=ORANGE,edgecolor=WHITE,interpolate=True, zorder=4)

ax.set_xlim([(1/(366*2)),0.013])
ax.set_ylim([-5,370])
ax.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])

ax.text(0.125, 0.7,'NDVI leads Precipitation', ha='center', va='center', rotation=0,transform=ax.transAxes)
ax.text(0.085, 0.225,'NDVI lags Precipitation', ha='center', va='center', rotation=0,transform=ax.transAxes)

ax.set_xlabel('Frequency (Hertz)')
ax.set_ylabel('Phase-Delay (days)')

p5= axvspan((1./(366+5)),(1./(366-5)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/2)),(1./((366.-5)/2)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/3)),(1./((366.-5)/3)), color=GRAY, alpha=0.5)
axvspan((1./((366.+5)/4)),(1./((366.-5)/4)), color=GRAY, alpha=0.5)

ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
ax2.set_xlabel('The Dependence of Phase-Delay on Frequency and Time (+ Phase Uncertainty Estimates via Monte Carlo Analysis)')
ax2.set_xticks([(1/366),(1./(366./2)),(1./(366./3)),(1./(366./4))])#0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
ax2.set_xticklabels([str(12),str(6),str(4),str(3)])
ax2.axis["right"].major_ticklabels.set_visible(False)

plt.legend([p2,p3,p4,p1,p5], ['phase-delay','single period','halve period','phase uncertainty estimates', 'frequencies of interest'])
plt.grid(True, zorder=0 )

plt.title('Time (months)')
plt.tight_layout()
out_pth = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia//phase_delay_montecarlo.png'
#plt.savefig(out_pth, dpi=200)
print out_pth
plt.show()

