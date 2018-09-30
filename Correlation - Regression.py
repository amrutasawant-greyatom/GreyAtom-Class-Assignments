
# coding: utf-8

# In[1]:


import os
os.chdir('E:\\Data Science\\datasets')


# In[51]:


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[19]:


df = pd.read_csv('train.csv')
df_copy = df
df.info()


# In[20]:


correlation_values = df.select_dtypes(include=[np.number]).corr()
correlation_values


# In[15]:


selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]


# In[16]:


selected_features


# In[29]:


# df[['GarageArea', 'SalePrice']]


# In[42]:


X = df[['OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'GarageCars']]


# In[43]:


y = df['SalePrice']


# In[44]:


from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression


# In[45]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)


# In[46]:


reg = LinearRegression()


# In[47]:


reg.fit(X_train, y_train)


# In[48]:


y_pred = reg.predict(X_test)


# In[49]:


reg.score(X_test, y_test)


# In[53]:


error = np.sqrt(mean_squared_error(y_test, y_pred))
error

