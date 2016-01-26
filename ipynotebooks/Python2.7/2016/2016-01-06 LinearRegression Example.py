
# coding: utf-8

# In[1]:

import numpy as np

# training data
X1=np.array([[-0.31994,-0.32648,-0.33264,-0.33844],[-0.32648,-0.33264,-0.33844,-0.34393],[-0.33264,-0.33844,-0.34393,-0.34913],[-0.33844,-0.34393,-0.34913,-0.35406],[-0.34393,-0.34913,-.35406,-0.35873],[-0.34913,-0.35406,-0.35873,-0.36318],[-0.35406,-0.35873,-0.36318,-0.36741],[-0.35873,-0.36318,-0.36741,-0.37144],[-0.36318,-0.36741,-0.37144,-0.37529],[-0.36741,-.37144,-0.37529,-0.37896],[-0.37144,-0.37529,-0.37896,-0.38069],[-0.37529,-0.37896,-0.38069,-0.38214],[-0.37896,-0.38069,-0.38214,-0.38349],[-0.38069,-0.38214,-0.38349,-0.38475],[-.38214,-0.38349,-0.38475,-0.38593],[-0.38349,-0.38475,-0.38593,-0.38887]])
X2=np.array([[-0.39265,-0.3929,-0.39326,-0.39361],[-0.3929,-0.39326,-0.39361,-0.3931],[-0.39326,-0.39361,-0.3931,-0.39265],[-0.39361,-0.3931,-0.39265,-0.39226],[-0.3931,-0.39265,-0.39226,-0.39193],[-0.39265,-0.39226,-0.39193,-0.39165],[-0.39226,-0.39193,-0.39165,-0.39143],[-0.39193,-0.39165,-0.39143,-0.39127],[-0.39165,-0.39143,-0.39127,-0.39116],[-0.39143,-0.39127,-0.39116,-0.39051],[-0.39127,-0.39116,-0.39051,-0.3893],[-0.39116,-0.39051,-0.3893,-0.39163],[-0.39051,-0.3893,-0.39163,-0.39407],[-0.3893,-0.39163,-0.39407,-0.39662],[-0.39163,-0.39407,-0.39662,-0.39929],[-0.39407,-0.39662,-0.39929,-0.4021]])

# target values
y1=np.array([-0.34393,-0.34913,-0.35406,-0.35873,-0.36318,-0.36741,-0.37144,-0.37529,-0.37896,-0.38069,-0.38214,-0.38349,-0.38475,-0.38593,-0.38887,-0.39184])
y2=np.array([-0.3931,-0.39265,-0.39226,-0.39193,-0.39165,-0.39143,-0.39127,-0.39116,-0.39051,-0.3893,-0.39163,-0.39407,-0.39662,-0.39929,-0.4021,-0.40506])


# In[124]:

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


# In[166]:

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
    #print 'training:',half_,'prediction:',half
    #half = to_forecast[0].shape[0]*0.5

    # do the work
    # apply linear regression
    LR_Multi = LinearRegression_Multi()
    # train first half of the data
    LR_Multi.fit(X[:,:half_], y[:,:half_])
    #print 'shape X training:',X[:,:half_].shape
    #print 'shape y training:',y[:,:half_].shape
    # predict second half of the data
    
    y_pred = LR_Multi.predict(X[:,half:])
    R2 = LR_Multi.scores(y_pred, y[:,half:])
    #print 'shape X prediction:',X[:,half:].shape
    #print 'shape y prediciton:',y[:,half:].shape
    
    return y_pred, R2


# In[75]:

import pandas as pd


# In[93]:

N = pd.read_csv(r'C:\Users\lenovo\Downloads//chart (1).csv', index_col='DateTime', parse_dates=True)
#test.sort_index(inplace=True)

#test.index = pd.date_range(np.datetime64(test.index[0]), np.datetime64(test.index[0]) + 
#                           np.timedelta64(test.shape[0]-1, 'D'), freq='d')
N.columns = ['VCI','TCI','VHI','NVAI','NTAI']
N = N.resample('d',how='mean')
N.interpolate(method='linear', order=3, inplace=True)
N.head()


# In[179]:

a = 0.5
NDAI = a * N.NVAI - (1 - a) * N.NTAI
NDAI.plot()

ewma_365 = pd.ewma(NDAI,span=365).plot()
ewma_182 = pd.ewma(NDAI,span=182).plot()
#dif = ewma_365 - ewma_182
#dif.plot()


# In[170]:

window = 7
horizon = 1
for i in [1,7,14,21,28]:
    idx_list = []
    r2_list = []
    for idx, window in enumerate(range(1,100)):
        y_pred, r2 = forecast(NDAI.as_matrix()[None],k=window,h=i,alpha=)
        y_pred = y_pred[0]
        r2 = r2[0]
        r2_list.append(r2)
        idx_list.append(idx+1)
    plt.plot(idx_list,r2_list)
plt.show()


# In[171]:

plt.plot(y_pred)


# In[158]:

range(1,30,7)


# In[121]:

y_pred = np.zeros_like(NDAI[:split])


# In[120]:

split, X.shape


# In[41]:

from sklearn.linear_model import LinearRegression

# train the 1st half, predict the 2nd half
half = len(y1)/2 # or y2 as they have the same length
LR = LinearRegression()
LR.fit(X, y)
pred = LR.predict(X)
r_2 = LR.score(X,y)
pred


# In[34]:

# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))


# In[58]:

plt.plot(y)
plt.plot(X)


# In[42]:

# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results
plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.plot(X, y_lin, c='r', label='Linear model')
plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.plot(X, pred, c='m', label='Linear2')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# In[71]:

y_new


# In[74]:

y_rbf = svr_lin.fit(X_new[0], y_new[0]).predict(X_new[0])
y_rbf


# In[50]:

k=7
h=1
to_forecast = X[None]
shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - k + 1, k)
print shape
strides = to_forecast.strides + (to_forecast.strides[-1],)
print strides
X_new = np.lib.stride_tricks.as_strided(to_forecast, shape=shape, strides=strides)
print X_new


# In[ ]:




# prepare a matrix y with the target values for each row of X
y = X[:,:,-1][:,h:]
X = X[:,:-h]


# In[ ]:




# In[ ]:




# In[27]:

from sklearn.svm import SVR


# In[28]:

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)


# In[32]:

y_lin = svr_lin.fit(X1[:half], y1[:half]).predict(X1[half:])
y_rbf = svr_rbf.fit(X1[:half], y1[:half]).predict(X1[half:])
y_rbf


# In[15]:

pred,y1[half:]


# In[19]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.scatter(pred,y1[half:])


# In[20]:

y_stack = np.vstack((y1[None],y2[None]))
X_stack = np.vstack((X1[None],X2[None]))


# In[21]:

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


# In[22]:

LR_Multi = LinearRegression_Multi()
LR_Multi.fit(X_stack[:,:half], y_stack[:,:half])
y_stack_pred = LR_Multi.predict(X_stack[:,half:])
R2 = LR_Multi.scores(y_stack_pred, y_stack[:,half:])


# In[24]:

import matplotlib.pyplot as plt
for i in range(len(R2)):
    plt.scatter(y_stack_pred[i],y_stack[i,half:])
    plt.show()


# In[25]:

R2


# In[ ]:



