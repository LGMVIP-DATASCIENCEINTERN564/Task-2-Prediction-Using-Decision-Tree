#!/usr/bin/env python
# coding: utf-8

# # LGMVIP Internship- Let's Grow More
# ## Data Science Internship task 2
# ## Prediction Using Decision Tree Algorithm:

# ### Import library

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.metrics import accuracy_score


# ### Loading Dataset

# In[3]:


df=pd.read_csv('Iris.csv')


# In[4]:


df.head()


# ### Dataset Info

# In[5]:


df.info()


# ### Finidng Null Values

# In[6]:


df.isnull().sum


# ### Drop Id columns:

# In[8]:


df=df.drop(columns=['Id'])
df.head()


# ### -Describe() is used to view some basic statistical details like percentile,mean,std etc. of a data frame of numeric values

# In[9]:


df.describe()


# In[12]:


df.shape


# In[13]:


df['Species'].value_counts()


# In[14]:


df['SepalLengthCm'].hist()


# In[15]:


df['SepalWidthCm'].hist()


# In[16]:


df['PetalLengthCm'].hist()


# In[17]:


df['PetalWidthCm'].hist()


# In[23]:


colors=['green','orange','blue']
species=['Iris-viriginca','Iris-versicolor','Iris-setosa']


# In[24]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.axis('equal')
l=['Versicolor','Setosa','Virginica']
s=[50,50,50]
ax.pie(s,labels=l,autopct='%1.2f%%')
plt.show()


# In[25]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[26]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[27]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalLengthCm'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[28]:


import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([df['SepalLengthCm']])
plt.figure(2)
plt.boxplot([df['SepalWidthCm']])
plt.show()


# In[29]:


df.plot(kind='density',subplots=True,layout=(3,3),sharex=False)


# In[30]:


df.plot(kind='box',subplots=True,layout=(2,5),sharex=False)


# In[31]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# In[32]:


sns.pairplot(df,hue='Species')


# In[33]:


df.corr()


# In[35]:


corr=df.corr()
fig, ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')


# In[36]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[37]:


df['Species']=le.fit_transform(df['Species'])
df.head()


# In[39]:


from sklearn.model_selection import train_test_split
X=df.drop(columns=['Species'])
Y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30)


# ### Model Building Decision Tree :

# In[40]:


clf=DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)


# In[41]:


clf


# In[42]:


y_pred=clf.predict(x_test)


# In[43]:


y_pred


# In[44]:


y_test


# In[46]:


data_frame =pd.DataFrame({'Actual Data':y_test,"Predicted Data":y_pred})


# In[47]:


data_frame.head()


# In[48]:


data_frame.tail()


# ### Accuracy of the model:

# In[50]:


print("Accuracy:",accuracy_score(y_test,y_pred))


# ### Visualizing the decision tree:

# In[51]:


col_names = ["Sepal length","Sepal width","Petal length","Petal width"]
target_names=["Setosa","Versicolor","Virginica"]


# In[52]:


plot_tree(clf,feature_names=col_names,class_names=target_names,filled=True)


# In[53]:


model=DecisionTreeClassifier().fit(X,Y)


# In[54]:


plt.figure(figsize=(20,15))
tree=tree.plot_tree(model,feature_names=col_names,class_names=target_names,filled=True)


# In[ ]:




