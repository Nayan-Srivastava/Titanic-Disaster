#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('T_train.csv')


# In[2]:


df.head()


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


df.describe(include='all')


# In[6]:


df['Survived'][df['Sex']=='female'].value_counts(normalize=True)[1]*100


# In[7]:


df['Survived'][df['Sex']=='male'].value_counts(normalize=True)[1]*100


# In[8]:


y_train=df['Survived']
X_train=df.drop(['PassengerId','Ticket','Cabin','Survived','Name'],axis=1)


# In[9]:


X_train.shape


# In[10]:


X_train.isnull().sum()


# In[11]:


mean=int(X_train['Age'].mean())


# In[12]:


X_train['Age'].fillna(mean,inplace=True)


# In[13]:


X_train.isnull().sum()


# In[14]:


X_train['Embarked'].value_counts()


# In[15]:


X_train['Embarked'].fillna('S',inplace=True)


# In[16]:


X_train.isnull().sum()


# In[17]:


X_train=pd.get_dummies(X_train)
X_train.shape


# In[18]:


X_train.head()


# In[33]:


import pandas as pd
df1=pd.read_csv('T_test.csv')


# In[34]:


df1.head()


# In[35]:


y_test=df1['Survived']
X_test=df1.drop(['PassengerId','Ticket','Cabin','Survived','Name'],axis=1)


# In[36]:


X_test.shape


# In[37]:


X_test.isnull().sum()


# In[38]:


m=int(X_test['Age'].mean())
X_test['Age'].fillna(m,inplace=True)
X_test.isnull().sum()


# In[39]:


X_test['Fare'].value_counts()


# In[40]:


me=int(X_test['Fare'].mean())
X_test['Fare'].fillna(me,inplace=True)


# In[41]:


X_test.isnull().sum()


# In[42]:


X_test=pd.get_dummies(X_test)
X_test.shape


# In[43]:


X_test.head()


# In[49]:


from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()

Lr.fit(X_train,y_train)


# In[46]:


from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)


# In[48]:


from sklearn.linear_model import Perceptron
p=Perceptron()
p.fit(X_train,y_train)


# In[51]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)


# In[52]:


from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
sgd.fit(X_train,y_train)


# In[53]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)


# In[54]:


print(Lr.score(X_test,y_test))
print(lsvc.score(X_test,y_test))
print(p.score(X_test,y_test))
print(rfc.score(X_test,y_test))
print(sgd.score(X_test,y_test))
print(gbc.score(X_test,y_test))


# In[55]:


y_predict=gbc.predict(X_test)


# In[56]:


df=pd.DataFrame({'PssengerId':df['PassengerId'],'Survived':y_predict })
df.to_csv("Titanic_output.csv")


# In[ ]:




