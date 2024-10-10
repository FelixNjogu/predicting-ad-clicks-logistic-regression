#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train=pd.read_csv('titanic_train.csv')


# In[5]:


train.head()


# In[7]:


train.count()


# In[10]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[11]:


sns.set_style('whitegrid')


# - Show number of survivers
# 

# In[18]:


sns.countplot(x='Survived',data=train,hue='Sex')


# In[20]:


sns.countplot(x='Survived',data=train,hue='Pclass')


# In[22]:


sns.countplot(data=train, x='SibSp')


# In[26]:


plt.figure(figsize=(10,4))
sns.boxplot(x='Pclass',y='Age',data=train)


# In[ ]:


def impute_age(cols):
    df1=df[['Pclass','Age']].groupby('Pclass').mean().reset_index()
    for i in df['Age']:
        if i is null:
            i= average(age)


# In[40]:


train[['Pclass','Age']].groupby('Pclass').mean().reset_index()


# In[41]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass ==1:
            return 38
        if Pclass ==2:
            return 29
        else:
            return 25
    else:
        return Age


# In[42]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[43]:


sns.heatmap(data=train.isnull(),yticklabels=False,cmap='viridis')


# In[47]:


# dropping the cabin column
train.drop('Cabin',axis=1,inplace=True)


# In[48]:


sns.heatmap(data=train.isnull(),yticklabels=False,cmap='viridis')


# In[51]:


#drop the remaining missing rows
train.dropna(inplace=True)
sns.heatmap(data=train.isnull(),yticklabels=False,cmap='viridis')


# In[55]:


#converting categorical features into numeric /dummy variables
pd.get_dummies(train['Sex'],drop_first=True)
sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[56]:


embark=pd.get_dummies(train['Embarked'],drop_first=True)
embark


# In[57]:


train=pd.concat([train,sex,embark],axis=1)


# In[58]:


train


# In[59]:


#drop columns that will not be used for prediction
train.drop(['Name','Sex','Ticket','PassengerId','Embarked'],axis=1,inplace=True)


# In[61]:


train.head()


# In[66]:


X=train.drop('Survived',axis=1)
y=train['Survived']


# In[64]:


from sklearn.model_selection import train_test_split


# In[68]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[69]:


from sklearn.linear_model import LogisticRegression


# In[75]:


logmodel=LogisticRegression(solver='lbfgs',max_iter=1000)


# In[76]:


logmodel.fit(X_train,y_train)


# In[78]:


predictions=logmodel.predict(X_test)


# In[79]:


predictions


# In[81]:


from sklearn.metrics import classification_report


# In[84]:


print(classification_report(y_test,predictions))


# In[ ]:




