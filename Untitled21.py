#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[3]:


df=pd.read_csv(r'C:\Users\hp\Downloads\taxi - Sheet1.csv')
df.head()


# In[7]:


x=df.iloc[:,0:-1]
x


# In[8]:


y=df.iloc[:,-1]


# In[10]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# In[23]:


from sklearn.linear_model import LinearRegression


# In[24]:


model=LinearRegression()
model.fit(x_train,y_train)


# In[25]:


model.score(x_test,y_test)



# In[27]:


pickle.dump(model,open('taxi.pkl','wb'))

model=pickle.load(open('taxi.pkl','rb'))
model.predict([[80,177000,6000,65]])
# In[31]:



# In[ ]:
