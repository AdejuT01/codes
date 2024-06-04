#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pickle


# In[2]:


df_train = pd.read_csv('wine_train.csv')
df_test = pd.read_csv('wine_test.csv')


# In[3]:


#Splitting the features and target variable
X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:, -1]

#Splitting the features and target variable
X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:, -1]


# In[4]:


from sklearn.preprocessing import LabelEncoder


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the data in the 'response' column
X_train['type'] = label_encoder.fit_transform(X_train['type'])
X_test['type'] = label_encoder.fit_transform(X_test['type'])


# In[5]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[6]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[8]:


pickle.dump(reg, open("model.pkl", "wb"))


# In[ ]:




