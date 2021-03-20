#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


df = pd.read_csv('Data/BankNote_Authentication.csv')


# In[3]:


df.head()


# In[4]:


# Dividing dataset into independent and dependent variables

X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[5]:


X.head()


# In[6]:


y.head()


# In[7]:


# Lets do the train test split

from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[9]:


#Implement random forest classifier

from sklearn.ensemble import RandomForestClassifier
Classifier=RandomForestClassifier()
Classifier.fit(X_train,y_train)


# In[10]:


# Lets check the predictions

y_pred=Classifier.predict(X_test)


# In[11]:


# Lets check the accuracy

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_pred)


# In[12]:


score


# In[14]:


# Create a pickle file using serilization

import pickle
pickle_out=open("classifier.pkl","wb")
pickle.dump(Classifier,pickle_out)
pickle_out.close()


# In[ ]:




