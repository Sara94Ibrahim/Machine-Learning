#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# load the dataset as pandas dataframe 
df01=pd.read_csv('Solubility_ESOL.csv')
df01.head(5)


# In[3]:


# to delete the columns: N, Compound ID, and ESOL predicted log(solubility: mol/L)
df02 = df01.drop(['N','Compound ID', 'ESOL predicted log(solubility:mol/L)'], axis =1)
df02.head()


# In[4]:


# To visualize the dataset distribution of each feature without standarization
plt.figure(figsize=(15,6))
sns.boxplot(data = df02)


# In[5]:


# to standarize the dataset (df02)
df_st =  StandardScaler().fit_transform(df02) 


# In[6]:


# to visualize the dataset after standardization
plt.figure(figsize=(15,6))
sns.boxplot(data = df_st)


# In[7]:


pca_out = PCA().fit(df_st)

# to get the variance in each of the PCs (from PC1 to PC11)
print(pca_out.explained_variance_ratio_)


# In[8]:


# to get the cumulative proportion of variance (from PC1 to PC11)   
print(np.cumsum(pca_out.explained_variance_ratio_))


# In[9]:


# to get component loadings (correlation coefficient between original variables and the PCs) 
loadings = pca_out.components_
num_pc = pca_out.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = df02.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[10]:


# to generate a correlation matrix plot for the loadings

fig, ax = plt.subplots(figsize=(8,8))  
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()


# In[11]:


# to visualize how much of the variance is explained by each PCs

df03 = pd.DataFrame({'var':pca_out.explained_variance_ratio_, 'PC':['PC1','PC2','PC3','PC4','PC5','PC6', 'PC7','PC8','PC9','PC10','PC11' ]})
sns.barplot(x='PC',y="var", data=df03, color="c")
plt.show()


# In[12]:


# to create the first user-defined function to generate the Loading Plot

def loadingplot(data, pca, width=5, height=5, margin=0.5):

    fig, ax = plt.subplots(figsize = (width,height))

    #set limits for figure
    x_min = min(pca.components_[0,:].min(),0)-margin
    x_max = max(pca.components_[0,:].max(),0)+margin
    y_min = min(pca.components_[1,:].min(),0)-margin
    y_max = max(pca.components_[1,:].max(),0)+margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    #scaling factor for text position
    text_pos = 0.05  #changed this

    for i, v in enumerate(pca.components_.T):
        ax.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.05, linewidth=2, color='red')#changed headwidth and headlength
        ax.text(v[0], v[1]+text_pos, data.columns[i], color='black', ha='center', va='center', fontsize=12)

    plt.plot([x_min, x_max], [0, 0], color='k', linestyle='--', linewidth=1)
    plt.plot([0, 0], [y_min, y_max], color='k', linestyle='--', linewidth=1)
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel("PC2", fontsize=14)
    ax.set_title("Loading plot", fontsize = 14)

    return ax


# In[13]:


# to appply the user-defined function with df02 as data 
ax2 = loadingplot(df02, pca_out, width=10, height=10, margin=0.2)
plt.show()


# In[14]:


# to create a datafame (df04) that will contain all PCs (from PC1 to PC11)
pca_scores = PCA().fit_transform(df_st)
df04 = pd.DataFrame(data = pca_scores,columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11'])

# to create a dataframe (df05) that will contain only PC1 and PC2
df05= df04[['PC1',"PC2"]]

# to create a dataframe (df_pc) that we will use to plot the Score Plot
df_pc = df05.join(df01)


# In[15]:


# to create the second user-defined function to generate the Score Plot

def scoreplot(df, parameter='TPSA'):
    fig = plt.gcf() 
    fig.set_size_inches(10, 8) 
    ax= sns.scatterplot(data= df, x='PC1', y='PC2', hue=parameter, palette='flare').set(title='Score Plot') 
    margin=0 
    x_min = min(df['PC1'].min(),0)-margin 
    x_max = max(df[ 'PC1'].max(),0)+margin 
    y_min = min(df['PC2'].min(),0)-margin 
    y_max = max(df['PC2'].max(),0)+ margin 

    plt.plot([x_min, x_max], [0, 0], color='k', linestyle='--', linewidth=1) 
    plt.plot([0, 0], [y_min, y_max], color='k', linestyle='--', linewidth=1) 
    return ax


# In[16]:


# to apply the second user-defined function with df_pc as data

ax2= scoreplot(df_pc, parameter='measured log(solubility:mol/L)')


# In[17]:


# to apply the second user-defined function with df_pc as data
#Visualising TPSA
ax2= scoreplot(df_pc, parameter='TPSA')


# In[18]:


ax2= scoreplot(df_pc, parameter='HeavyAtomCount')


# In[ ]:




