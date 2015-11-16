# TEST WITH THIS URL:
# http://localhost/cgi-bin/pywps.cgi?request=Execute&identifier=PyWCS_Cross_Spectral&service=WPS&version=1.0.0
# coding: utf-8
# In[1]:
from __future__ import division
import types
from pywps.Process import WPSProcess
import urllib
from lxml import etree
from datetime import datetime, timedelta
import jdcal
import pandas as pd
import json
import logging
import numpy as np
import nitime.algorithms as tsa
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import cStringIO
from PIL import Image
#get_ipython().magic(u'matplotlib inline')

# --------*-*-*-*-*-*--------  WPSProcess  --------*-*-*-*-*-*--------  WPSProcess  --------*-*-*-*-*-*-----
# Computing diagonal for each row of a 2d array. See: http://stackoverflow.com/q/27214027/2459096
def makediag3d(M):
    b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
    b[:, ::M.shape[1] + 1] = M
    
    logging.info('function `makediag3d` complete')    
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

    logging.info('function `get_starter_matrix` complete')
    return mat

def HANTS(sample_count, inputs,
          frequencies_considered_count=3,
          outliers_to_reject='Lo',
          low=0., high=255,
          fit_error_tolerance=5,
          delta=0.1):
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
    """

    # define some parameters
    base_period_len = sample_count  #

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
        
        # multiply outliers with timeseries
        za = np.einsum('ijk,ik->ij', mat, p * inputs)
        #print za

        # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
        diag = makediag3d(p)
        #print diag
        
        A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
        # add delta to suppress high amplitudes but not for [0,0]
        A += Adelta
        #A[:, 0, 0] = A[:, 0, 0] - delta
        #print A

        # solve linear matrix equation and define reconstructed timeseries
        zr = np.linalg.solve(A, za)
        #print zr
        
        outputs = np.einsum('ijk,kj->ki', mat.T, zr)
        #print outputs

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

    logging.info('function `HANTS` complete')
    return outputs

# In[2]:

def convert_ansi_date(date, offset=0.5):
    logging.info('function `convert_ansi_date` complete')
    return jdcal.jd2gcal(2305812.5, date + offset) # 0.5 offset is to adjust from night to noon

def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    logging.info('function `unix_time` complete')
    return delta.total_seconds()    

def unix_time_millis(dt):
    logging.info('function `unix_time_millis` complete')
    return int(unix_time(dt) * 1000)

def anomaly_computation(ts, norm=1):
    if norm == 2:
        anom_ts = ts.groupby([ts.index.month, ts.index.day]).apply(lambda g: (g - g.mean())/(g.max()-g.min()))
    if norm == 1:
        anom_ts = ts.groupby([ts.index.month, ts.index.day]).apply(lambda g: g - g.mean())
    if norm == 0:
        anom_ts = ts.groupby([ts.index.month, ts.index.day]).apply(lambda g: (g - g.mean())/g.std())
    return anom_ts


# In[3]:

# --------*-*-*-*-------  Compute multitaper + uncertainty -------*-*-*-*--------
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
    #print ('x size', x.shape)
    #print ('y size', y.shape)
    
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
    #print ('iteration no is', mc_no)
    
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
    
# repeat or pead signals number of times
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

def plotCrossSpectral(f,ph,coh,phif,tile_no):
    vspan_start = 0.0026
    vspan_end = 0.0028
    xlim_start = 0.000
    xlim_end = 0.012
    
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
    
    # Save to string and compress adaptive
    ram = cStringIO.StringIO()    
    plt.savefig(ram, dpi=90)#bbox_inches='tight', pad_inches=0, dpi=90, transparent=True)
    plt.show()
    plt.close()    
    ram.seek(0)
    im = Image.open(ram)
    im2 = im.convert('RGB', palette=Image.ADAPTIVE)
    output = cStringIO.StringIO()
    im2.save(output, format="PNG")
    return output

def plotCrossSpectral2(t,x,y,fkx,fky,cxy,f,ph,coh,phif):
    
    vspan_start = 0.0026
    vspan_end = 0.0028
    xlim_start = 0.000
    xlim_end = 0.012
    
    plt.figure(figsize=(14,7.5))

    ## PLOT 1
    plt.subplot(331)
    plt.grid(axis='y')

    plt.plot(t,x, 'k-', lw=1, label='TRMM')
    plt.plot(t,y, 'k--', lw=1, label='NDVI')

    plt.xlim(0,364)
    #plt.ylim(-0.2,0.2)
    plt.yticks([-0.2,-0.1,0,0.1,0.2])
    plt.ylabel('normalised anomaly')
    plt.xlabel('day of year')
    plt.title('anomaly timeseries')
    leg = plt.legend(loc=1, frameon=True, ncol=2)
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
    # Save to string and compress adaptive
    ram = cStringIO.StringIO()    
    plt.savefig(ram, bbox_inches='tight', pad_inches=0, dpi=100, transparent=True)
    plt.show()
    plt.close()    
    ram.seek(0)
    im = Image.open(ram)
    im2 = im.convert('RGBA', palette=Image.ADAPTIVE)
    output = cStringIO.StringIO()
    im2.save(output, format="PNG")
    return output  

class Process(WPSProcess):
    def __init__(self):
        
        WPSProcess.__init__(self,
             identifier       = "PyWCS_Cross_Spectral", 
             title            = "***---*** Cross Spectral Computation ***---***",
             version          = "1",
             storeSupported   = "true",
             statusSupported  = "false",
             abstract         = "This service provide access to compute the Cross Spectrum")
              
        self.latitude  = self.addLiteralInput(identifier  = "lat_input",
                                              title       = "Input Latitude",
                                              type        = types.FloatType,
                                              default     = "31.812")

        self.longitude  = self.addLiteralInput(identifier  = "long_input",
                                               title       = "Input Longitude",
                                               type        = types.FloatType,
                                               default     = "-8.742")

        self.ansistart  = self.addLiteralInput(identifier  = "ansi_start_input",
                                               title       = "Input ANSI Start",
                                               type        = types.IntType,
                                               default     = "146098")

        self.ansiend  = self.addLiteralInput(identifier  = "ansi_end_input",
                                             title       = "Input ANSI End",
                                             type        = types.IntType,
                                             default     = "150481")

        self.coverageidprec  = self.addLiteralInput(identifier  = "coverageid_precipitation_input",
                                                    title       = "Input CoverageID Precipitation",
                                                    type        = types.StringType,
                                                    default     = "trmm_3b42_coverage_1")
  
        self.coverageidndvi  = self.addLiteralInput(identifier  = "coverageid_ndvi_input",
                                                    title       = "Input CoverageID NDVI",
                                                    type        = types.StringType,
                                                    default     = "modis_13c1_cov")

        self.yeartoanalyze  = self.addLiteralInput(identifier  = "year_to_analyze_input",
                                                   title       = "Input Year To Analyze",
                                                    type        = types.StringType,
                                                    default     = "2005")

        self.mc_no  = self.addLiteralInput(identifier  = "monte_carlo_input",
                                           title       = "Input Number of Monte Carlo Iterations",
                                           type        = types.IntType,
                                           default     = "10")

        self.ti_no  = self.addLiteralInput(identifier  = "tiling_input",
                                           title       = "Input Number of Tiling",
                                           type        = types.IntType,
                                           default     = "50")

        self.exten  = self.addLiteralInput(identifier  = "extend_input",
                                           title       = "Input How To Extend",
                                           type        = types.IntType,
                                           default     = "0")

        self.prwin  = self.addLiteralInput(identifier  = "precipitation_window",
                                           title       = "Set window for precipitation summation",
                                           type        = types.IntType,
                                           default     = "10")

        self.stand  = self.addLiteralInput(identifier  = "AnomalyStandardization_cspec",
                                           title       = "Compute absolute or standardized anomaly",
                                           type        = types.IntType,
                                           default     = "0")

        self.nfreq  = self.addLiteralInput(identifier  = "no_freq",
                                           title       = "Number of frequencies to consider",
                                           type        = types.IntType,
                                           default     = "32")

        self.hants  = self.addLiteralInput(identifier  = "based_on_hants_cspec",
                                           title       = "NDVI based on HANTS yes or no",
                                           type        = types.IntType,
                                           default     = "0")

        self.outli  = self.addLiteralInput(identifier  = "outlier_type",
                                           title       = "Which type of values are considered as outliers",
                                           type        = types.IntType,
                                           default     = "0")
                                            
        self.resu1 = self.addComplexOutput(identifier  = "cross_spectrum_output", 
                                           title       = "Output Cross-Spectrum Analyses PNG",
                                           formats     = [{'mimeType':'image/png'}]) #xml/application :: text/xml

         #self.resu2 = self.addComplexOutput(identifier  = "AnomalyNDVI_ewma", 
         #                                   title       = "Output NDVI anomaly expx. weighted moving average",
         #                                   formats     = [{'mimeType':'text/xml'}]) #xml/application

    def execute(self):  


        # WCS Endpoint
        endpoint= 'http://159.226.117.95:8080/rasdaman/ows'

        # Coordinates + Timestamp
        lat_input = float(self.latitude.getValue())#31.812
        long_input = float(self.longitude.getValue())#-8.742
        start_date_input = int(self.ansistart.getValue())#146098
        end_date_input = int(self.ansiend.getValue())#150481

        # CoverageID's
        coverageid_precipitation = str(self.coverageidprec.getValue())#'trmm_3b42_coverage_1'
        coverageid_ndvi = str(self.coverageidndvi.getValue())#'modis_13c1_cov'

        # Year to analyze
        year_to_analyze = str(self.yeartoanalyze.getValue())#'2005'

        # Cross-spectrum settings
	precip_window = int(self.prwin.getValue())#10
        mc_no = int(self.mc_no.getValue())#10
        tile_no = int(self.ti_no.getValue())#50
        extend = int(self.exten.getValue())#0
        standard = int(self.stand.getValue())#0

        # HANTS settings
        based_on_hants = int(self.hants.getValue())#1
        frequencies = int(self.nfreq.getValue())#32
        variable_out = int(self.outli.getValue())#1
        if variable_out == 1:
            outliers = "Hi"    
        elif variable_out == 2:
            outliers = "Lo"
        else:
            outliers = "None" 

        # # Create URL for TRMM based on the separated parameters
        # In[6]:

        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']=coverageid_precipitation#'modis_13c1_cov'
        field['SUbSET']='Lat('+str(lat_input)+','+str(lat_input)+')'
        field['SUBSET']='Long('+str(long_input)+','+str(long_input)+')'
        field['SuBSET']='ansi('+str(start_date_input)+','+str(end_date_input)+')'


        # In[7]:
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        data_trmm = urllib.urlopen(full_url).read()
        root_trmm = etree.fromstring(data_trmm)

        # In[8]:
        # read grid envelope of domain set
        xml_low_env = root_trmm[1][0][0][0][0].text
        xml_high_env = root_trmm[1][0][0][0][1].text

        # load grid envelope as numpy array
        low_env = np.array((root_trmm[1][0][0][0][0].text.split(' '))).astype(int)
        high_env = np.array((root_trmm[1][0][0][0][1].text.split(' '))).astype(int)
        ts_shape = high_env - low_env + 1

        easting = ts_shape[0]
        northing = ts_shape[1]
        time = ts_shape[2]


        # In[9]:

        # extract the values we need from the parsed XML
        sta_date_ansi = int((root_trmm.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text.split(' '))[2]) #146098
        end_date_ansi = int((root_trmm.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text.split(' '))[2]) #150481
        sta_date_rasd = int((root_trmm.find(".//{http://www.opengis.net/gml/3.2}low").text.split(' '))[2]) #146098
        end_date_rasd = int((root_trmm.find(".//{http://www.opengis.net/gml/3.2}high").text.split(' '))[2]) #150480
        timestep_date = int((root_trmm[1][0][5].text.split(' '))[2]) #1

        # compute the start and end-date
        dif_date_anra = sta_date_ansi - sta_date_rasd
        dif_date_rasd = end_date_rasd - sta_date_rasd + 1
        end_date_anra = dif_date_rasd * timestep_date + sta_date_rasd + dif_date_anra

        sd = convert_ansi_date(sta_date_ansi) # (2001, 1, 1, 0.5)
        ed = convert_ansi_date(end_date_anra) # (2013, 1, 1, 0.5)

        # convert dates to pandas date_range
        str_date = str(sd[1])+'.'+str(sd[2])+'.'+str(sd[0])+'.'+str(int(np.round(sd[3]*24)))+':00'
        end_date = str(ed[1])+'.'+str(ed[2])+'.'+str(ed[0])+'.'+str(int(np.round(ed[3]*24)))+':00'
        freq_date = str(int(timestep_date))+'D'
        dates = pd.date_range(str_date,end_date, freq=freq_date)
        dates_trmm = dates[:-1]

        logging.info('dates converted from ANSI to ISO 8601')


        # In[10]:

        # read data block of range set
        # load data block as numpy array
        ts = np.array((root_trmm[2][0][1].text.split(','))).astype(float)
        ts_reshape_trmm = ts.reshape((easting*northing,time)) #Easting = ts_shape[0], Northing = ts_shape[1], time = ts_shape[2]



        # In[11]:

        df = pd.DataFrame(ts_reshape_trmm[0],dates_trmm)
        df = df.ix['2000':'2010-09']

        df_daly = df.asfreq('D', method='pad')
        df_10sum = pd.rolling_sum(df_daly, precip_window)

        df_anom_trmm = anomaly_computation(df_10sum, norm=standard)#standard


        # # Adapt URL for NDVI based on the separated parameters

        # In[12]:

        field['COVERAGEID']='modis_13c1_cov'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        data_ndvi = urllib.urlopen(full_url).read()
        root_ndvi = etree.fromstring(data_ndvi)


        # In[13]:

        # read grid envelope of domain set
        xml_low_env = root_ndvi[1][0][0][0][0].text
        xml_high_env = root_ndvi[1][0][0][0][1].text

        # load grid envelope as numpy array
        low_env = np.array((root_ndvi[1][0][0][0][0].text.split(' '))).astype(int)
        high_env = np.array((root_ndvi[1][0][0][0][1].text.split(' '))).astype(int)
        ts_shape = high_env - low_env + 1

        easting = ts_shape[0]
        northing = ts_shape[1]
        time = ts_shape[2]


        # In[14]:

        # extract the values we need from the parsed XML
        sta_date_ansi = int((root_ndvi.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text.split(' '))[2]) #146095
        end_date_ansi = int((root_ndvi.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text.split(' '))[2]) #150495
        sta_date_rasd = int((root_ndvi.find(".//{http://www.opengis.net/gml/3.2}low").text.split(' '))[2]) #9131
        end_date_rasd = int((root_ndvi.find(".//{http://www.opengis.net/gml/3.2}high").text.split(' '))[2]) #9405
        timestep_date = int((root_ndvi[1][0][5].text.split(' '))[2]) #16

        # compute the start and end-date
        dif_date_anra = sta_date_ansi - sta_date_rasd
        dif_date_rasd = end_date_rasd - sta_date_rasd + 1
        end_date_anra = dif_date_rasd * timestep_date + sta_date_rasd + dif_date_anra

        sd = convert_ansi_date(sta_date_ansi) # (2000, 12, 29, 0.5)
        ed = convert_ansi_date(end_date_anra) # (2013, 1, 15, 0.5)

        # convert dates to pandas date_range
        str_date = str(sd[1])+'.'+str(sd[2])+'.'+str(sd[0])+'.'+str(int(np.round(sd[3]*24)))+':00'
        end_date = str(ed[1])+'.'+str(ed[2])+'.'+str(ed[0])+'.'+str(int(np.round(ed[3]*24)))+':00'
        freq_date = str(int(timestep_date))+'D'
        dates = pd.date_range(str_date,end_date, freq=freq_date)
        dates_ndvi = dates[:-1]

        logging.info('dates converted from ANSI to ISO 8601')


        # In[15]:
        # read data block of range set
        # load data block as numpy array
        #ts = np.array((root_ndvi[2][0][1].text.split(','))).astype(float)
        #ts_reshape_ndvi = ts.reshape((easting*northing,time)) #Easting = ts_shape[0], Northing = ts_shape[1], time = ts_shape[2]

        # In[16]:
        # read data block of range set
        xml_ts = cStringIO.StringIO(root_ndvi[2][0][1].text)
         
        # load data block as numpy array
        ts = np.loadtxt(xml_ts, dtype='float', delimiter=',')
        #ts = ts[0:-1]
        #ts_reshape = ts.reshape((easting*northing,time)) #Easting = ts_shape[0], Northing = ts_shape[1], time = ts_shape[2]

        # In[8]:
        if based_on_hants == 1:
            ts_reshape_ndvi=ts[None] 
            ts_reshape_ndvi = HANTS(sample_count=ts_reshape_ndvi.shape[1], inputs=ts_reshape_ndvi/100, frequencies_considered_count=frequencies,  outliers_to_reject=outliers)
            ts_reshape_ndvi = ts_reshape_ndvi[0]*100
        if based_on_hants == 0:
            ts_reshape_ndvi = ts


        df = pd.DataFrame(ts_reshape_ndvi/1000.,dates_ndvi)
        df = df.ix['2000':'2010-09']

        df_daly = df.asfreq('D', method='pad')

        df_anom_ndvi = anomaly_computation(df_daly, norm=standard)


        # In[17]:

        year1=year_to_analyze
        year2=year_to_analyze
        #df_anom_ndvi.ix[year1:year2].plot()
        #df_anom_trmm.ix[year1:year2].plot()

        ndvi_data = df_anom_ndvi.ix[year1:year2]
        trmm_data = df_anom_trmm.ix[year1:year2]

        ndvi_data1 = ndvi_data - ndvi_data.mean()
        trmm_data1 = trmm_data - trmm_data.mean()

        #ndvi_data1.plot()
        #trmm_data1.plot()

        # # Computation of the cross-spectrum
        # In[18]:

        x,y,t = pad_extend(m1=trmm_data1.values.flatten(),m2=ndvi_data1.values.flatten(), tile_no=tile_no, extend=extend)
        f, fkx, fky, cxy, ph, coh = mtem(x,y)
        phif = mtem_unct(x,y,coh, mc_no=mc_no)

        # In[19]:
        #PlotAsString = plotCrossSpectral(f,ph,coh,phif,tile_no)
        PlotAsString = plotCrossSpectral2(t,x,y,fkx,fky,cxy,f,ph,coh,phif)

        # ouput
        self.resu1.setValue(PlotAsString)
