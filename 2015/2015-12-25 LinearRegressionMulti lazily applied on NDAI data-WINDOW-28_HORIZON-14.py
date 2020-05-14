
# coding: utf-8

# In[1]:

import numpy as np
from osgeo import gdal
import os
import sys


# In[2]:

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


# In[3]:

def listall(RootFolder, wildcard='', extension='.tif'):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith(extension)]
    return lists


# In[4]:

class LinearRegression_Multi:
    def stacked_lstsq(self, L, b, rcond=1e-10):
        """
        Solve L x = b, via SVD least squares cutting of small singular values
        L is an array of shape (..., M, N) and b of shape (..., M).
        Returns x of shape (..., N)
        """
        u, s, v = np.linalg.svd(L, full_matrices=False)
        s_max = s.max(axis=-1, keepdims=True)
        s_min = rcond*s_max
        inv_s = np.zeros_like(s)
        inv_s[s >= s_min] = 1/s[s>=s_min]
        x = np.einsum('...ji,...j->...i', v,
                      inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
        return np.conj(x, x)    
    
    def center_data(self, X, y):
        """ Centers data to have mean zero along axis 0. 
        """
        # center X        
        X_mean = np.average(X,axis=1)
        X_std = np.ones(X.shape[0::2])
        X = X - X_mean[:,None,:] 
        # center y
        y_mean = np.average(y,axis=1)
        y = y - y_mean[:,None]
        return X, y, X_mean, y_mean, X_std

    def set_intercept(self, X_mean, y_mean, X_std):
        """ Calculate the intercept_
        """
        self.coef_ = self.coef_ / X_std # not really necessary
        self.intercept_ = y_mean - np.einsum('ij,ij->i',X_mean,self.coef_)

    def scores(self, y_pred, y_true ):
        """ 
        The coefficient R^2 is defined as (1 - u/v), where u is the regression
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        sum of squares ((y_true - y_true.mean()) ** 2).sum().        
        """        
        u = ((y_true - y_pred) ** 2).sum(axis=-1)
        v = ((y_true - y_true.mean(axis=-1)[None].T) ** 2).sum(axis=-1)
        r_2 = 1 - u/v
        return r_2
    
    def fit(self,X, y):
        """ Fit linear model.        
        """         
        # get coefficients by applying linear regression on stack
        X_, y, X_mean, y_mean, X_std = self.center_data(X, y)
        self.coef_ = self.stacked_lstsq(X_, y)
        self.set_intercept(X_mean, y_mean, X_std)

    def predict(self, X):
        """Predict using the linear model
        """
        return np.einsum('ijx,ix->ij',X,self.coef_) + self.intercept_[None].T


# In[5]:

def forecast(to_forecast, k=7, h=1, alpha=0.5):
    """
    Forecast method using linear regresion
    Input
    to_forecast: input of matrix to forecast
    k          : forecast window, number of previous observation to use
                 default 7 (one week)
    h          : forecast horizon, number of periods ahead to forecast
                 default 1 (one day)
    alpha      : default 0.5. Use first half to train, 2nd half to predict
                 0.1: use first 10% to train and remaining 90% to predict
                 0.9: use first 90% to train and remaining 10% to predict                  
    
    Output
    y_pred     : matrix containing the predictions
    R2         : matrix containing the R2 values
    """
    # prepare a matrix X where each row contains a forecast window
    shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - k + 1, k)
    strides = to_forecast.strides + (to_forecast.strides[-1],)
    X = np.lib.stride_tricks.as_strided(to_forecast, shape=shape, strides=strides)
    
    # prepare a matrix y with the target values for each row of X
    y = X[:,:,-1][:,h:]
    X = X[:,:-h]
    
    # training data is rounded down, prediction data is rounded up
    half_ = int(np.floor(to_forecast[0].shape[0]*alpha))
    half  = int(np.ceil(to_forecast[0].shape[0]*(1-alpha)))
    print 'training:',half_,'prediction:',half
    #half = to_forecast[0].shape[0]*0.5

    # do the work
    # apply linear regression
    LR_Multi = LinearRegression_Multi()
    # train first half of the data
    LR_Multi.fit(X[:,:half_], y[:,:half_])
    print 'shape X training:',X[:,:half_].shape
    print 'shape y training:',y[:,:half_].shape
    # predict second half of the data
    
    y_pred = LR_Multi.predict(X[:,half:])
    R2 = LR_Multi.scores(y_pred, y[:,half:])
    print 'shape X prediction:',X[:,half:].shape
    print 'shape y prediciton:',y[:,half:].shape
    
    return y_pred, R2


# In[6]:

path_base = r'D:\Data\LS_DATA\NDAI-1day_IM_bbox_warp//NDAI_2003_001_IM_bbox_wrap.tif'
folder_ndai = r'D:\Data\LS_DATA\NDAI-1day_IM_bbox_warp'


# In[7]:

# register all of the GDAL drivers
gdal.AllRegister()

# open the image
ds = gdal.Open(path_base)
if ds is None:
    print ('Could not open base file')
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

files_ndai = listall(folder_ndai)
noFiles = len(files_ndai)
Files = np.array([len(files_ndai)])
print ( noFiles ) 


# In[ ]:

#base_ = np.zeros(shape=(len_,rows,cols), dtype=np.float32)

# loop through the rows
for i in range(0, rows, yBlockSize): #(0, rows, yBlockSize)
    if i + yBlockSize < rows:
        numRows = yBlockSize
    else:
        numRows = rows - i
    
    # loop through the columns
    for j in range(0, cols, xBlockSize):# (0, cols, xBlockSize)
        if j + xBlockSize < cols:
            numCols = xBlockSize
        else:
            numCols = cols - j
        
        print ('column',j,'row',i,'numCols',numCols,'numRows',numRows)
        # set base array to fill 
        ap_ndai = np.zeros(shape=(noFiles,numRows,numCols), dtype=np.float32)        

        #select blocks from trmm and ndvi files
        for m, file_ndai in enumerate(files_ndai):
            try:
                raster = gdal.Open(file_ndai, gdal.GA_ReadOnly)
                band = raster.GetRasterBand(1)
                ap_ndai[m] = band.ReadAsArray(j, i, numCols, numRows).astype(np.float)
            except:
                print m,file_ndai

        #reshape from 3D to 2D
        ap_ndai_2D = ap_ndai.reshape((noFiles,numRows*numCols)).T
        ap_ndai_2D = np.nan_to_num(ap_ndai_2D)

        # prepare output prediction array
        window = 28 # forecast window (last 7 days)
        horizon = 14 # forecast horizon (1 day)
        alpha = 0.5
        split = int(np.ceil(ap_ndai_2D[0].shape[0]*(1-alpha)) - (window + horizon))
        y_pred = np.zeros_like(ap_ndai_2D[:,:split])
        R2 = np.zeros((numRows,numCols)).ravel()
        # prepare prediction array
        #y_pred = np.zeros_like(ap_ndai_2D[:,int(math.ceil(ap_ndai_2D[0].shape[0]*(1-alpha))):])        
        #hants_trmm = np.zeros_like(ap_ndai_2D)

        # apply linear regression prediction lazily with maximum 10000 rows each time
        for k in range(0, (numRows*numCols), 10000):
            if k + 10000 < (numRows*numCols):
                l = k + 10000
            else:
                l = k + (numRows*numCols) - k
            print ('start',k,'end',l,ap_ndai_2D[k:l].shape)

            y_pred[k:l], R2[k:l]  = forecast(ap_ndai_2D[k:l], window, horizon, alpha)

        # reshape from 2D to 3D
        y_pred = y_pred.T.reshape(split,numRows,numCols)
        R2 = R2.reshape(numRows,numCols)
        
        # save 3D blocks to temp
        folder_temp_y_pred = r'D:\tmp\ndai\ndai\window28horizon14'
        folder_temp_r2 = r'D:\tmp\ndai\R2\window28horizon14'
        file_temp_y_pred = 'Y_PRED_col_'+str(j).zfill(4)+'_row_'+str(i).zfill(4)+'_numCols_'+str(numCols).zfill(4)+'_numRows_'+str(numRows).zfill(4)
        file_temp_r2 = 'R2_col_'+str(j).zfill(4)+'_row_'+str(i).zfill(4)+'_numCols_'+str(numCols).zfill(4)+'_numRows_'+str(numRows).zfill(4)
        path_temp_y_pred = folder_temp_y_pred + '//' + file_temp_y_pred
        path_temp_r2 = folder_temp_r2 + '//' + file_temp_r2
        print path_temp_y_pred
        print path_temp_r2
        np.save(path_temp_y_pred, y_pred)
        np.save(path_temp_r2, R2)


# In[ ]:

# load 2D tiles from temp and save as tif
folder_temp = r'D:\tmp\ndai\R2\window28horizon14'
files_temp = listall(folder_temp, extension='.npy')
noFilesTemp = len(files_temp)
print ( noFilesTemp ) 

base_ = np.zeros(shape=(rows,cols), dtype=np.float32)
#for i in range(noFiles):
#    print (i)
for j in files_temp:
    print (j)
    numRows = int(j[-8:-4])
    numCols = int(j[-21:-17])    
    row = int(j[-34:-30])
    col = int(j[-43:-39])
    
    
    print ('col',col,'row',row,'numCols',numCols,'numRows',numRows)

    load_temp_tile = np.load(j,mmap_mode='r')
    base_[row:row+numRows,col:col+numCols] = load_temp_tile[:,:]

folder_out = r'D:\tmp\ndai_out\R2\window28horizon14//'
file_out = 'NDAI_R2_.tif'
path_out = folder_out+file_out
print (path_out)
saveRaster(path_out,base_,ds,datatype=6)


# In[8]:

# prepare output prediction array
window = 28 # forecast window (last 7 days)
horizon = 14 # forecast horizon (1 day)
alpha = 0.5
split = int(np.ceil(Files*(1-alpha)) - (window + horizon))
filename_out = files_ndai[(noFiles - split):]


# In[17]:

filename_out[0][-25:-14]


# In[ ]:




# In[18]:


# load 2D tiles from temp and save as tif
folder_temp = r'D:\tmp\ndai\ndai\window28horizon14'
files_temp = listall(folder_temp, extension='.npy')
noFilesTemp = len(files_temp)
print ( noFilesTemp ) 

base_ = np.zeros(shape=(rows,cols), dtype=np.float32)
for i in range(noFiles):
    print (i)
    for j in files_temp:
        print (j)
        numRows = int(j[-8:-4])
        numCols = int(j[-21:-17])    
        row = int(j[-34:-30])
        col = int(j[-43:-39])


        print ('col',col,'row',row,'numCols',numCols,'numRows',numRows)

        load_temp_tile = np.load(j,mmap_mode='r')
        base_[row:row+numRows,col:col+numCols] = load_temp_tile[i,:,:]

    print '\n'
    folder_out = r'D:\tmp\ndai_out\NDAI\window28horizon14//'
    file_out = 'NDAI_R2_'+filename_out[i][-25:-17]+'.tif'
    path_out = folder_out+file_out
    print (path_out)
    saveRaster(path_out,base_,ds,datatype=6)


# In[ ]:




# In[ ]:



