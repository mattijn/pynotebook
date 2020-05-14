
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

import numpy as np

# training data
X1=np.array([[-0.31994,-0.32648,-0.33264,-0.33844],[-0.32648,-0.33264,-0.33844,-0.34393],[-0.33264,-0.33844,-0.34393,-0.34913],[-0.33844,-0.34393,-0.34913,-0.35406],[-0.34393,-0.34913,-.35406,-0.35873],[-0.34913,-0.35406,-0.35873,-0.36318],[-0.35406,-0.35873,-0.36318,-0.36741],[-0.35873,-0.36318,-0.36741,-0.37144],[-0.36318,-0.36741,-0.37144,-0.37529],[-0.36741,-.37144,-0.37529,-0.37896],[-0.37144,-0.37529,-0.37896,-0.38069],[-0.37529,-0.37896,-0.38069,-0.38214],[-0.37896,-0.38069,-0.38214,-0.38349],[-0.38069,-0.38214,-0.38349,-0.38475],[-.38214,-0.38349,-0.38475,-0.38593],[-0.38349,-0.38475,-0.38593,-0.38887]])
X2=np.array([[-0.39265,-0.3929,-0.39326,-0.39361],[-0.3929,-0.39326,-0.39361,-0.3931],[-0.39326,-0.39361,-0.3931,-0.39265],[-0.39361,-0.3931,-0.39265,-0.39226],[-0.3931,-0.39265,-0.39226,-0.39193],[-0.39265,-0.39226,-0.39193,-0.39165],[-0.39226,-0.39193,-0.39165,-0.39143],[-0.39193,-0.39165,-0.39143,-0.39127],[-0.39165,-0.39143,-0.39127,-0.39116],[-0.39143,-0.39127,-0.39116,-0.39051],[-0.39127,-0.39116,-0.39051,-0.3893],[-0.39116,-0.39051,-0.3893,-0.39163],[-0.39051,-0.3893,-0.39163,-0.39407],[-0.3893,-0.39163,-0.39407,-0.39662],[-0.39163,-0.39407,-0.39662,-0.39929],[-0.39407,-0.39662,-0.39929,-0.4021]])

# target values
y1=np.array([-0.34393,-0.34913,-0.35406,-0.35873,-0.36318,-0.36741,-0.37144,-0.37529,-0.37896,-0.38069,-0.38214,-0.38349,-0.38475,-0.38593,-0.38887,-0.39184])
y2=np.array([-0.3931,-0.39265,-0.39226,-0.39193,-0.39165,-0.39143,-0.39127,-0.39116,-0.39051,-0.3893,-0.39163,-0.39407,-0.39662,-0.39929,-0.4021,-0.40506])
half = len(y1)/2 # or y2 as they have the same length


# In[ ]:

# def slow_lstsq(L, b):
#     return np.array([np.linalg.lstsq(L[k], b[k])[0]
#                      for k in range(L.shape[0])])


# In[3]:

def stacked_lstsq(L, b, rcond=1e-10):
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


# In[ ]:

from sklearn.linear_model import LinearRegression

# train the 1st half, predict the 2nd half

regressor = LinearRegression()
regressor.fit(X2[:half], y2[:half])
pred = regressor.predict(X2[half:])
r_2 = regressor.score(X2[half:],y2[half:])

# print the prediction and r^2
print 'pred:',pred
print 'r^2:',r_2


# In[ ]:

# set inner variables used to center data and get intercept
fit, X_mean, y_mean, X_std = regressor.fit(X2[:half], y2[:half])
intercept = y_mean - np.dot(X_mean, regressor.coef_)
# apply prediction
npdot = np.dot(X2[half:],regressor.coef_)
prediction = npdot + intercept


# In[ ]:

print 'y_mean:', y_mean, y_mean.shape
print 'X_mean:', X_mean, X_mean.shape
print 'coef_:', regressor.coef_, regressor.coef_.shape
print 'npdot:', npdot, npdot.shape
print 'intercept:', intercept, intercept.shape
print 'predict:', prediction, prediction.shape


# In[4]:

# stack X1 & X2 and y1 & y2 
y_stack = np.vstack((y1[None],y2[None]))
X_stack = np.vstack((X1[None],X2[None]))

print 'y1 shape:',y1.shape, 'X1 shape:',X1.shape
print 'y_stack shape:',y_stack.shape, 'X_stack:',X_stack.shape


# In[5]:

# center X_stack
X_stack_mean = np.average(X_stack[:,:half],axis=1)
X_stack_std = np.ones(X_stack[:,:half].shape[0::2])
X_stack_center = X_stack[:,:half] - X_stack_mean[:,None,:]
#X_stack -= X_stack_mean[:,None,:]

# center y_stack
y_stack_mean = np.average(y_stack[:,:half],axis=1)
y_stack_center = y_stack[:,:half] - y_stack_mean[:,None]
#y_stack -= y_stack_mean[:,None]


# In[6]:

y_stack_center


# In[ ]:

# get coefficients by applying linear regression on stack
coef_stack = stacked_lstsq(X_stack_center, y_stack_center)
print 'coef_stack:',coef_stack


# In[ ]:

# calculate the intercept
coef_stack = coef_stack / X_stack_std
intercept_stack = y_stack_mean - np.einsum('ij,ij->i',X_stack_mean,coef_stack)
print 'intercept_stack:',intercept_stack


# In[ ]:

# apply prediction using einsum
einsum_stack = np.einsum('ijx,ix->ij',X_stack[:,half:],coef_stack)#X_stack[:,:half]
print 'einsum:',einsum_stack
print 'npdot:',npdot
prediction_stack = einsum_stack + intercept_stack[None].T
print 'prediction_stack:',prediction_stack
y_stack_true = y_stack[:,half:]
print 'y_stack_true:',y_stack_true


# In[ ]:

#The coefficient R^2 is defined as (1 - u/v), where u is the regression
#sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
#sum of squares ((y_true - y_true.mean()) ** 2).sum().
u = ((y_stack_true - prediction_stack) ** 2).sum(axis=-1)
v = ((y_stack_true - y_stack_true.mean(axis=-1)[None].T) ** 2).sum(axis=-1)
r_2_stack = 1 - u/v


# In[ ]:



