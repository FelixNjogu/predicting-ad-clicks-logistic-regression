#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


ad_data=pd.read_csv('advertising.csv')


# In[4]:


ad_data.head()


# In[5]:


ad_data.info()


# In[8]:


ad_data['City'].value_counts()


# In[10]:


ad_data.Country.nunique()


# In[20]:


#to see columns with null rows
ad_data.info()


# In[16]:


plt.figure(figsize=(10,4))
sns.histplot(data=ad_data,x='Age',color='orange')


# In[26]:


plt.figure(figsize=(10,4))
sns.histplot(data=ad_data,x='Daily Time Spent on Site',binwidth=5)


# In[21]:


#statistical description
ad_data.describe()


# In[24]:


sns.pairplot(data=ad_data, hue='Clicked on Ad', palette='viridis')


# In[27]:


from sklearn.model_selection import train_test_split


# In[30]:


X=ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y=ad_data['Clicked on Ad']


# In[31]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


model=LogisticRegression()
model.fit(X_train,y_train)


# In[34]:


predictions=model.predict(X_test)


# In[42]:


from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,predictions))


# In[41]:


print(confusion_matrix(y_test,predictions))

