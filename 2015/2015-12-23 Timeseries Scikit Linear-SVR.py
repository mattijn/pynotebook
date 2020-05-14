
# coding: utf-8

# In[14]:

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
get_ipython().magic(u'matplotlib inline')


# In[24]:

data = pd.read_csv(r'D:\Downloads\Mattijn@Zhou\anomaly_class/revenue-example.csv', sep=",")
week     = data['week'][:, np.newaxis]
revenue  = data['revenue']


# In[25]:

data


# In[30]:

lr = LinearRegression()
train_week = week[0:5]
train_revenue = revenue[0:5]
lr.fit(train_week, train_revenue)

b_0   = lr.intercept_
coeff = lr.coef_


# In[31]:

# Let's just test some points.
pred_week = week[5::]
pred_rev  = revenue[5::]


# In[37]:

# Let's predict the values for existing weeks (Testing)
pred = lr.predict(pred_week)

plt.scatter(week, revenue, color='g')
plt.scatter(train_week, train_revenue, color='b')
plt.scatter(pred_week, pred, color='red')
plt.show()


# In[38]:

# Not query pretty, but we align our week matrices.
predict_week = np.array(
  [a for a in xrange(max(train_week)+1, max(train_week)+3)]
)[:, np.newaxis]


# In[39]:

predict_week


# In[40]:

forecast_2w = lr.predict(predict_week)
forecast_2w


# In[21]:

print forecast_2w

forecast_2w[0] == b_0 + (coeff * 12) // np.array([ True], dtype=bool)
forecast_2w[1] == b_0 + (coeff * 13) // np.array([ True], dtype=bool)


# In[23]:

# sklearn has an r2_score method. 
score = r2_score(revenue, lr.predict(week[:]))
print score

# Or you can `score` the one from LinearRegression
score = lr.score(week[:], revenue[:])
print score


# In[ ]:



