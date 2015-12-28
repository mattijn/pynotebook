
# coding: utf-8

# In[1]:

import numpy as np


# In[174]:

# from sklearn.linear_model import LinearRegression
# LR = LinearRegression()
# y_pred = []
# R2 = []
# def loopLR(X_stack, y_stack, LR):
#     for idx in range(y_stack.shape[0]):    
#         LR.fit(X_stack[idx,:half], y_stack[idx,:half])
#         y_pred_ = LR.predict(X_stack[idx,half:])
#         R2_ = LR.score(X_stack[idx,half:],y_stack[idx,half:])
#         y_pred.append(y_pred_)
#         R2.append(R2_)    
#     return R2, y_pred
# loopLR(X_stack,y_stack,LR)


# In[142]:

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


# In[143]:

# training data
X1=np.array([[-0.31994,-0.32648,-0.33264,-0.33844],[-0.32648,-0.33264,-0.33844,-0.34393],[-0.33264,-0.33844,-0.34393,-0.34913],[-0.33844,-0.34393,-0.34913,-0.35406],[-0.34393,-0.34913,-.35406,-0.35873],[-0.34913,-0.35406,-0.35873,-0.36318],[-0.35406,-0.35873,-0.36318,-0.36741],[-0.35873,-0.36318,-0.36741,-0.37144],[-0.36318,-0.36741,-0.37144,-0.37529],[-0.36741,-.37144,-0.37529,-0.37896],[-0.37144,-0.37529,-0.37896,-0.38069],[-0.37529,-0.37896,-0.38069,-0.38214],[-0.37896,-0.38069,-0.38214,-0.38349],[-0.38069,-0.38214,-0.38349,-0.38475],[-.38214,-0.38349,-0.38475,-0.38593],[-0.38349,-0.38475,-0.38593,-0.38887]])
X2=np.array([[-0.39265,-0.3929,-0.39326,-0.39361],[-0.3929,-0.39326,-0.39361,-0.3931],[-0.39326,-0.39361,-0.3931,-0.39265],[-0.39361,-0.3931,-0.39265,-0.39226],[-0.3931,-0.39265,-0.39226,-0.39193],[-0.39265,-0.39226,-0.39193,-0.39165],[-0.39226,-0.39193,-0.39165,-0.39143],[-0.39193,-0.39165,-0.39143,-0.39127],[-0.39165,-0.39143,-0.39127,-0.39116],[-0.39143,-0.39127,-0.39116,-0.39051],[-0.39127,-0.39116,-0.39051,-0.3893],[-0.39116,-0.39051,-0.3893,-0.39163],[-0.39051,-0.3893,-0.39163,-0.39407],[-0.3893,-0.39163,-0.39407,-0.39662],[-0.39163,-0.39407,-0.39662,-0.39929],[-0.39407,-0.39662,-0.39929,-0.4021]])

# target values
y1=np.array([-0.34393,-0.34913,-0.35406,-0.35873,-0.36318,-0.36741,-0.37144,-0.37529,-0.37896,-0.38069,-0.38214,-0.38349,-0.38475,-0.38593,-0.38887,-0.39184])
y2=np.array([-0.3931,-0.39265,-0.39226,-0.39193,-0.39165,-0.39143,-0.39127,-0.39116,-0.39051,-0.3893,-0.39163,-0.39407,-0.39662,-0.39929,-0.4021,-0.40506])

half = len(y1)/2 # or y2 as they have the same length
# stack X1 & X2 and y1 & y2 
y_stack = np.vstack((y1[None],y2[None]))
X_stack = np.vstack((X1[None],X2[None]))


# In[147]:

LR_Multi = LinearRegression_Multi()


# In[177]:

LR_Multi.fit(X_stack[:,:half], y_stack[:,:half])
y_stack_pred = LR_Multi.predict(X_stack[:,half:])
R2_stack = LR_Multi.scores(y_stack_pred, y_stack[:,half:])


# In[ ]:



