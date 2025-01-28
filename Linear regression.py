#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# load the dataset as a pandas data frame 
df01 = pd.read_csv('Solubility_ESOL.csv')

# to remove the unnecessary columns 
df02 = df01.drop(['N','Compound ID', 'ESOL predicted log(solubility:mol/L)','measured log(solubility:mol/L)',], axis =1)

# To standardize df02
from sklearn.preprocessing import StandardScaler 
df_st = StandardScaler().fit_transform(df02)

# to create a new dataframe(df_feat) with the standardized features 
df_feat = pd.DataFrame(df_st, columns = df02.columns) 
df_feat = df01[['measured log(solubility:mol/L)']].join(df_feat)
df_feat.head()


# In[9]:


from sklearn.model_selection import train_test_split
X = df_feat.drop(['measured log(solubility:mol/L)'], axis= 1)
y = df_feat[['measured log(solubility:mol/L)']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[10]:


from sklearn.linear_model import LinearRegression
model_lr01 = LinearRegression()
model_lr01.fit(X_train, y_train)


# In[11]:


# to see the model's coefficients
df_model_coef = pd.DataFrame(model_lr01.coef_, columns=[X_train.columns])
df_model_coef = df_model_coef.T
df_model_coef.columns=['Coefficients']
df_model_coef


# In[12]:


#to see the model's intercept
model_lr01.intercept_


# In[ ]:


#help(LinearRegression)


# In[13]:


pred = model_lr01.predict(X_test)
df_res = pd.DataFrame(data = pred, columns=["pred"])
df_res


# In[14]:


array_y_test = np.array(y_test)
df_res['y_test']=pd.DataFrame(data=array_y_test)
df_res


# In[15]:


sns.scatterplot(data=df_res, x= 'pred', y='y_test')
plt.xlabel('pred: Predicted log(solubility:mol/L)')
plt.ylabel('y_test: Measured log(solubility:mol/L)')


# In[16]:


#to generate the residual data
df_res['residual']= df_res['y_test']- df_res['pred']
df_res


# In[17]:


# to generate the residual plot
sns.residplot(data=df_res, x='pred',y='residual')


# In[18]:


# to visualize the distribution of the residial data  
sns.displot(data= df_res, x='residual',bins=40, kde=True)


# In[19]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE = mean_absolute_error(y_test,pred)
MSE = mean_squared_error(y_test,pred)
RMSE = np.sqrt(MSE)
print('MAE',"%.4f"% MAE, sep='=')
print('MSE',"%.4f"% MSE, sep='=')
print('RMSE',"%.4f"% RMSE, sep='=')


# ## 2.Multiple Linear Regression with Statsmodel 

# In[1]:


conda install -c conda-forge statsmodels


# In[23]:


df03 = y_train.join(X_train)
df03 = df03.rename(columns={"measured log(solubility:mol/L)": "measured"})
df03


# In[24]:


import statsmodels.api as sm
no_consider = df03.columns.difference(['measured'])
model_lr02 = sm.OLS.from_formula( 'measured ~' + '+'.join(no_consider), data=df03)


# In[25]:


result2 = model_lr02.fit()
print (result2.summary())


# In[26]:


no_consider = df03.columns.difference(['measured','FractionCSP3','HeavyAtomCount', 'MolWt','TPSA' ])
model_lr03 = sm.OLS.from_formula( 'measured ~' + '+'.join(no_consider), data=df03)
result3 = model_lr03.fit()
print (result3.summary())


# In[27]:


pred3 = result3.predict(X_test)
df_res3 = pd.DataFrame(data = pred3, columns=["pred3"])
df_res3 = df_res3.join(y_test)
df_res3 = df_res3.rename(columns={"measured log(solubility:mol/L)": "y_test"})
df_res3['residual']= df_res3['y_test']- df_res3['pred3']
df_res3


# In[28]:


MAE = mean_absolute_error(y_test,pred3)
MSE = mean_squared_error(y_test,pred3)
RMSE = np.sqrt(MSE)
print('MAE',"%.4f"% MAE, sep='=')
print('MSE',"%.4f"% MSE, sep='=')
print('RMSE',"%.4f"% RMSE, sep='=')

