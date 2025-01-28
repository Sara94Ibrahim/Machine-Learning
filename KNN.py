#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#load the dataset as a pandas dataframe
df01 = pd.read_csv('breast_cancer_prediction.csv', delimiter = ",")
df01.head()


# In[7]:


df02=df01.drop(['diagnosis', 'id', 'Unnamed: 32'], axis =1)
df02.head()


# In[8]:


#to visualse the dataset distribution of each feature without standardisation
plt.figure(figsize=(15,6))
ax = sns.boxplot(data = df02)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90);


# In[10]:


#to standardise the data frame (df02) and create a new one (df_st)
from sklearn.preprocessing import StandardScaler
df_st = StandardScaler().fit_transform(df02)


# In[11]:


#to visualse the dataset after standardisation
plt.figure(figsize=(15,6))
sns.boxplot(data = df_st)


# In[13]:


#to create a new dataframe (df_feat) with the standardised features
df_feat = pd.DataFrame(df_st, columns = df02.columns)
df_feat.head()


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[15]:


X = df_feat
y = df01['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# In[19]:


knn = KNeighborsClassifier(n_neighbors =1)
knn.fit(X_train, y_train)


# In[20]:


pred = knn.predict(X_test)


# In[21]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[22]:


#to see the y_test data
ax = sns.countplot(x = y_test, label = 'count')
ax.bar_label(ax.containers[0])


# In[24]:


# to create a user-defined function that will generate a more detailed confusion matrix
def conf_matrix(real_test, prediction):
    cm = confusion_matrix(real_test, prediction)
    plt.figure(figsize=(4,1))
    ax = sns.set(font_scale =1.1)
    x_labels = ["B", 'M']
    y_labels = ["B", 'M']
    fig = sns.heatmap(cm, annot = True, cmap = 'Blues', fmt = 'g', cbar = False, xticklabels = x_labels, yticklabels = y_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True Labels")
    plt.show(fig)
    return ax

#end 


# In[25]:


#to apply the user-defined function
conf_matrix(y_test, pred)


# In[26]:


error_rate = []
for i in range (1, 40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i !=y_test))
    
plt.figure(figsize = (10, 6))
plt.plot(range(1, 40), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o',
        markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate Vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[28]:


error_rate = []
for i in range (1, 10):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i !=y_test))
    
plt.figure(figsize = (10, 6))
plt.plot(range(1, 10), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o',
        markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate Vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[29]:


knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(classification_report(y_test, pred))


# In[30]:


conf_matrix(y_test, pred) #the first model is better I guess

