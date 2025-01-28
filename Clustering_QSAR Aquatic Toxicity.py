#!/usr/bin/env python
# coding: utf-8

# ## K-means Clustering

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = "3"  # Set the environment variable


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[3]:


df = pd.read_csv('qsar_aquatic_toxicity.csv', sep = ';')
df


# In[4]:


plt.clf()
fig = plt.figure(figsize = (7,7), dpi = 100)
axes1 = fig.add_axes([0.5, 0.5, 0.5, 0.5])
axes1.scatter(df['LC50'], df['MLOGP'])
axes1.set_xlabel('LC50 (-Log mol/L)')
axes1.set_ylabel('LogP')
axes1.set_title('Lethal Concentration (LC50) Vs LogP')
plt.show()


# In[5]:


X = df[['LC50','MLOGP']].to_numpy()
X


# In[6]:


kmeans_2_clusters = KMeans(n_clusters = 2, n_init = 10, random_state = 123)
kmeans_2_clusters.fit(X)


# In[7]:


print(kmeans_2_clusters.labels_)


# In[8]:


plt.clf()
plt.figure(figsize = (5,5), dpi = 100)
plt.scatter(X[:,0], X[:,1], s = 50, c = kmeans_2_clusters.labels_, cmap = plt.cm.bwr)
plt.scatter(kmeans_2_clusters.cluster_centers_[:,0],
            kmeans_2_clusters.cluster_centers_[:,1],
            marker = '*',
            s = 350,
            color = 'black',
            label = 'Centers')
plt.legend(loc = 'best')
plt.xlabel('LC50 (-Log mol/L)')
plt.ylabel('LogP')
plt.title('K-means: 2 clusters')
plt.show()


# In[9]:


kmeans_3_clusters = KMeans(n_clusters = 3, n_init = 10, random_state = 123)
kmeans_3_clusters.fit(X)


# In[10]:


print(kmeans_3_clusters.labels_)


# In[11]:


plt.clf()
plt.figure(figsize = (5,5), dpi = 100)
plt.scatter(X[:,0], X[:,1], s= 50, c= kmeans_3_clusters.labels_, cmap = plt.cm.prism)
plt.scatter(kmeans_3_clusters.cluster_centers_[:,0],
            kmeans_3_clusters.cluster_centers_[:,1],
            marker = '*',
            s = 350,
            color = 'black',
            label = 'Centers')
plt.legend(loc = 'best')
plt.xlabel('LC50 (-Log mol/L)')
plt.ylabel('LogP')
plt.title('K-means: 3 clusters')
plt.show()


# In[12]:


X = df[['LC50', 'MLOGP']].to_numpy()
X


# In[13]:


km_out_single_run = KMeans(n_clusters = 4, n_init = 1, random_state = 123).fit(X)
print(km_out_single_run.inertia_)


# In[14]:


km_out_single_run = KMeans(n_clusters = 4, n_init = 10, random_state = 123).fit(X)
print(km_out_single_run.inertia_)


# In[15]:


km_out_single_run = KMeans(n_clusters = 4, n_init = 50, random_state = 123).fit(X)
print(km_out_single_run.inertia_)


# In[16]:


plt.clf()
inertia = []
ninit = []
for i in range (1,50):
    km_out_single_run = KMeans(n_clusters = 4, n_init = i, random_state = 123).fit(X)
    inertia.append(km_out_single_run.inertia_)
    ninit.append(i)
    print(km_out_single_run.inertia_)
plt.plot(ninit, inertia)
plt.xlabel('n_init')
plt.ylabel('inertia')
plt.show()


# In[17]:


plt.clf()
inertia = []
ninit = []
for i in range (1,5):
    km_out_single_run = KMeans(n_clusters = 4, n_init = i, random_state = 123).fit(X)
    inertia.append(km_out_single_run.inertia_)
    ninit.append(i)
    print(km_out_single_run.inertia_)
plt.plot(ninit, inertia)
plt.xlabel('n_init')
plt.ylabel('inertia')
plt.show()


# ## Hierarchial Clustering

# In[18]:


from scipy.cluster.hierarchy import linkage
hc_complete = linkage(X, "complete") #maximim distance between objects #complete must not be in caps


# In[19]:


hc_average = linkage(X, "average") #mean distance between objects
hc_single = linkage(X, "single") #minimum distance between objects


# In[20]:


from scipy.cluster.hierarchy import dendrogram

#to calculate the complete linkage dendrogram
plt.figure(figsize =(30, 10))
plt.title('HCD: Complete Linkage (maximum distance)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
dendrogram(
    hc_complete, truncate_mode = 'level', p=6,
    leaf_rotation =90., #rotates the x axis labels
    leaf_font_size = 8., #font size for the x axis labels
)
plt.show()


# In[21]:


from scipy.cluster.hierarchy import dendrogram

#to calculate the complete linkage dendrogram
plt.figure(figsize =(30, 10))
plt.title('HCD: Complete Linkage (maximum distance)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
dendrogram(
    hc_complete, truncate_mode = 'level', p=10,
    leaf_rotation =90., #rotates the x axis labels
    leaf_font_size = 8., #font size for the x axis labels
)
plt.show()


# In[22]:


from scipy.cluster.hierarchy import cut_tree
print(cut_tree(hc_complete, n_clusters = 2).T)


# In[23]:


#To calculate the average linkage dendrogram
plt.figure(figsize=(30,10))
plt.title('HCD: Average Linkage(mean distance)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
dendrogram(
    hc_average,truncate_mode = 'level', p=9,
    leaf_rotation=90., #rotates the x axis labels
    leaf_font_size=8., #font size for the x axis labels
)
plt.show()


# In[24]:


print(cut_tree(hc_average, n_clusters = 2).T)


# In[25]:


#To calculate the single linkage dendrogram
plt.figure(figsize=(30, 10))
plt.title('HCD: Single Linkge(minimum distance)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
dendrogram(
    hc_single,truncate_mode='level', p=20,
    leaf_rotation=90., #rotates the x axis labels
    leaf_font_size=8., #font size for the x axis labels
)
plt.show()


# In[ ]:




