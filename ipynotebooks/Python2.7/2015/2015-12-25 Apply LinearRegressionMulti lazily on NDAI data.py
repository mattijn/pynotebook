
# coding: utf-8

# In[ ]:

import numpy as np
from osgeo import gdal
import os


# In[ ]:

def listall(RootFolder, wildcard='', extension='.tif'):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith(extension)]
    return lists


# In[ ]:

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


# In[ ]:

path_base = r'J:\NDAI_2003-2014\2003//NDAI_2003_001.tif'
folder_ndai = r'J:\NDAI_2003-2014'


# In[ ]:

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
        #ap_ndai = np.zeros(shape=(noFiles,numRows,numCols), dtype=np.float32)        
        
        # select blocks from trmm and ndvi files
        #for m in range(noFiles):
            #raster = gdal.Open(files_ndai[m], gdal.GA_ReadOnly)
            #band = raster.GetRasterBand(1)
            #ap_ndai[m] = band.ReadAsArray(j, i, numCols, numRows).astype(np.float)
               
        # reshape from 3D to 2D
        #ap_ndai_2D = ap_ndai.reshape((noFiles,numRows*numCols)).T
        #ap_ndai_2D *= 1000

#         # prepare HANTS
#         hants_trmm = np.zeros_like(ap_ndai_2D)

#         # apply HANTS lazily with maximum 100 rows each time
#         for k in range(0, (numRows*numCols), 100):
#             if k + 100 < (numRows*numCols):
#                 l = k + 100
#             else:
#                 l = k + (numRows*numCols) - k
#             print ('start',k,'end',l,ap_trmm_2D[k:l].shape)
            
#             hants_trmm[k:l] = HANTS(noFiles, ap_trmm_2D[k:l], frequencies_considered_count=6, outliers_to_reject='None', low=-1000, high=1000)
        
#         # reshape from 2D to 3D
#         hants_trmm /= 1000
#         hants_trmm = hants_trmm.T.reshape(noFiles,numRows,numCols)
        
#         # save 3D blocks to temp
#         folder_temp = r'D:\tmp\ndai'
#         file_temp = 'col_'+str(j).zfill(3)+'_row_'+str(i).zfill(3)+'_numCols_'+str(numCols).zfill(3)+'_numRows_'+str(numRows).zfill(3)
#         path_temp = folder_temp + '//' + file_temp
#         np.save(path_temp, hants_trmm)


# In[ ]:

# load 2D tiles from temp and save as tif
folder_temp = r'D:\tmp'
files_temp = listall(folder_temp, extension='.npy')
noFilesTemp = len(files_temp)
print ( noFilesTemp ) 

base_ = np.zeros(shape=(rows,cols), dtype=np.float32)
for i in range(noFiles):
    print (i)
    for j in files_temp:
        print (j)
        col = int(j[-39:-36])
        row = int(j[-31:-28])
        numCols = int(j[-19:-16])
        numRows = int(j[-7:-4])
        print ('col',col,'row',row,'numCols',numCols,'numRows',numRows)
        
        load_temp_tile = np.load(j,mmap_mode='r')
        base_[row:row+numRows,col:col+numCols] = load_temp_tile[i,:,:]
        
        folder_out = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\Jiujang\TRMM2006_DryYear\10_Day_Period\10_DaySums_StdNormAnomalyResClipHANTS//'
        file_out = 'TRMM_JJ_2006'+str(i+1).zfill(3)+'.tif'
        path_out = folder_out+file_out
        print (path_out)
        saveRaster(path_out,base_,ds,datatype=6)

