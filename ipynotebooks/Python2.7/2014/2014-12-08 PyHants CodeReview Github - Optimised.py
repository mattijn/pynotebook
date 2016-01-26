
# coding: utf-8

# In[1]:

get_ipython().magic(u'load_ext line_profiler')
get_ipython().magic(u'load_ext memory_profiler')
#from mattijn import *


# In[2]:

#!/usr/local/env python
import numpy as np

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


# import profilehooks
# @profilehooks.profile(sort='time')
def HANTS(sample_count, inputs,
          frequencies_considered_count=3,
          outliers_to_reject='Hi',
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


# In[3]:

y = array_in(10)
# Fit to high values, set low values as outliers
plt.figure(figsize=(15,3))
y_out = HANTS(26, y, 3, outliers_to_reject='Lo')

plt.subplot(121)
plt.plot(y_out[0], label='Low')

plt.subplot(122)
plt.plot(y_out[4], label='Low')

y_out = HANTS(26, y, 3, outliers_to_reject='None')
plt.subplot(121)
plt.plot(y_out[0], label='None')

plt.subplot(122)
plt.plot(y_out[4], label='None')

y_out = HANTS(26, y, 3, outliers_to_reject='Hi')
plt.subplot(121)
plt.scatter(np.arange(26),y[0])
plt.plot(y_out[0], label='High')
plt.ylim(0)
plt.xlim(0,35)
plt.grid()
plt.legend()

plt.subplot(122)
plt.scatter(np.arange(26),y[4], c='r')
plt.plot(y_out[4], label='High')
plt.ylim(0)
plt.xlim(0,35)
plt.legend()
plt.grid()
plt.show()


# In[4]:

y.shape


# In[242]:

arrlen = [5,10,50,100,500,1000,5000,10000]
arrlen = np.asarray(arrlen)
for i in range(len(arrlen)):
    y = array_in(arrlen[i]) 
    get_ipython().magic(u'timeit HANTS(26,y)')


# In[264]:

result_1 = np.array([0.00251, 0.0033, 0.0107, 0.0238, 0.109, 0.247, 1.37, 2.76])
result_2 = result_1 / arrlen
result_2


# In[115]:

from timeit import Timer, timeit, repeat
arrlen = [5,10,50,100,500,1000,5000,10000]#256,512,1024,2048,4096,8192,16384,32768]
#arrlen = [8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
rep_num = 100


# In[116]:

arrlen = np.asarray(arrlen)
result = np.zeros((arrlen.shape[0],rep_num))

for i in range(len(arrlen)):
    y = array_in(arrlen[i])
    result[i] = repeat("HANTS(26, y, 3, outliers_to_reject='Lo')", setup="from __main__ import HANTS, y", repeat=rep_num, number=1)
               
    print (arrlen[i],np.min(result[i]))


# In[276]:

import numpy as np
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
plt.grid()
t = arrlen
s1 = result_2 * 1000
s1[7] = 0.299
ax1.plot(t, s1, 'k-')
ax1.plot(t, s1, 'ko', markersize=5)
#ax1.scatter(t, s1, 'b-')
ax1.set_xlabel('size of input array')
ax1.set_xscale('log')
ax1.set_ylim(0.2,0.6)
ax1.set_yticks([0.2,0.3,0.4,0.5,0.6])
ax1.set_xlim(arrlen[0],arrlen[-1])
y_lab = np.min((result.T/arrlen).T)*1000
ax1.annotate('214us', xy=(50, s1.min()),  xycoords='data',
            xytext=(35, 35), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=10"))

# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('computation time for a single array (ms)', color='k')
for tl in ax1.get_yticklabels():
    tl.set_color('k')


ax2 = ax1.twinx()
s2 = result_1

ax2.plot(t, s2, 'm')
ax2.plot(t, s2, 'mo', markersize=5)
ax2.set_yscale('linear')
#ax2.set_ylim(-0.1,4.5)
ax2.set_xlim(arrlen[0],arrlen[-1])
#ax2.scatter(t, s2, 'r-')
ax2.set_ylabel('computation time for total array (s)', color='m')

ax2.annotate(str(s2.max())+'s', xy=(10**4, s2.max()),  xycoords='data',
            xytext=(-50, 10), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=10"))


for tl in ax2.get_yticklabels():
    tl.set_color('m')

plt.savefig(r'C:\Users\lenovo\Documents\HOME\Figures AJ//pyhants.png', dpi=300)
plt.show()

