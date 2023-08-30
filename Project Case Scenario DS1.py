#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[6]:


boston_df.head


# In[8]:


plt.figure(figsize=(8,6))
sns.boxplot(data = boston_df, y='MEDV')

plt.ylabel('Median Value of Homes')
plt.title('Boxplot of Median Value of Owner-Occupied Homes')

plt.show()


# In[9]:


plt.figure(figsize=(6, 4))
sns.countplot(data = boston_df, x ='CHAS')

plt.xlabel('Charles River')
plt.ylabel('Count')
plt.title('Bar Plot of Charles River')
plt.show()


# In[10]:


boston_df['AGE.group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, float('inf')],
                               labels=['35 years and younger', 'between 35 and 70 years', '70 years and older'])

plt.figure(figsize=(8, 6))
sns.boxplot(boston_df, x='AGE.group', y='MEDV')

plt.xlabel('Age Group')
plt.ylabel('Median Value of Homes')
plt.title('Boxplot of Median Value of Homes vs Age Group')
plt.show()


# In[11]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=boston_df, x='NOX', y='INDUS')

plt.xlabel('Nitric Oxide Concentrations')
plt.ylabel('Proportion of Non-Retail Business Acres')
plt.title('Scatter Plot: Nitric Oxide vs Proportion of Non-Retail Business Acres')

plt.show()


# In[13]:


plt.figure(figsize=(8, 6))
sns.histplot(data=boston_df, x='PTRATIO', bins=20)

plt.xlabel('Pupil-to-Teacher Ratio')
plt.ylabel('Frequency')
plt.title('Histogram of Pupil-to-Teacher Ratio')
plt.show()


# In[14]:


df = boston_df

group_a = df[df['CHAS'] == 1]['MEDV']
group_b = df[df['CHAS'] == 0]['MEDV']

t_statistic, p_value = stats.ttest_ind(group_a, group_b)
alpha = 0.05

if p_value < alpha:
    print("There is significant difference in median value.")
else:
    print("There is no significant difference in median value.")


# In[16]:


f_statistic, p_value = stats.f_oneway(*[group['MEDV'] for _, group in df.groupby('AGE')])
alpha = 0.05

if p_value < alpha:
    print("There is significant difference in median value of houses for different age groups.")
else:
    print("There is no significant difference in median value of houses for different age groups.")


# In[17]:


correlation_coefficient, p_value = stats.pearsonr(df['NOX'], df['INDUS'])
alpha = 0.05

if p_value < alpha:
    print("There is a significant relationship between Nitric Oxide concentration and non-retail business acres.")
else:
    print("There is no significant relationship between Nitric Oxide concentration and non-retail business acres.")


# In[18]:


X = df['DIS']
y = df['MEDV']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())


# In[ ]:




