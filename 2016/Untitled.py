
# coding: utf-8

# In[14]:

def leap_year(year):
    if year % 4 == 0 and year %100 != 0 or year % 400 == 0:
        out = 1        
    else:
        out = 0
    return out


# In[21]:

for i in range(1,366+leap_year(2000)):
    print i


# In[16]:

leap_year(2004)


# In[8]:

for i in m:
    print i
    print 'end'


# In[13]:

[x * x for x in range(10)]


# In[ ]:



