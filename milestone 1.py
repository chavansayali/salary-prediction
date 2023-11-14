#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Importing necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import missingno as msno
plt.rcParams['figure.figsize'] = (8,6)

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Loading the dataset


# In[3]:


data=pd.read_csv('salarydata.csv')
data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.isna().sum()


# In[7]:


data.drop(['capital-gain','capital-loss','education-num'],axis=1,inplace=True)
data.head()


# In[8]:


#Exploratory Data Analysis (EDA)


# In[9]:


cat_cols = data.columns[data.dtypes == 'object']
cat_cols


# In[10]:


for cols in cat_cols:
    print(data[cols].value_counts(), '\n')


# Found '?' in columns occupation, workclass, and native-country. Replace this using mode

# In[11]:


data[(data == '?').any(axis = 1)]


# In[12]:


data['workclass'].mode()


# In[13]:


data['occupation'].mode()


# In[14]:


data['native-country'].mode()


# In[15]:


# Replacing '?' with the mode values for each column
data['workclass'] = data['workclass'].replace('?','Private')
data['occupation']=data['occupation'].replace('?','Prof-speciality')
data['native-country']=data['native-country'].replace('?','United-States')


# In[16]:


sns.pairplot(data)
plt.show()


# In[17]:


data.hist(figsize=(15,6))
plt.show()


# In[18]:


#We can check the above using some representations like graphs, barplots, etc

#checking age

plt.hist(data['age'])
plt.xlabel('Age')
plt.ylabel('Count')
# plt.xticks(np.arange(20,100,5))
# plt.rcParams['figure.figsize'] = (8,8)
plt.title('Age Distribution')
plt.show()


# In[19]:


#checking workclass

sns.countplot(x = data['workclass'], data = data)
plt.xticks(rotation = 90)
plt.show()


# In[20]:


#checking education

data['education'].value_counts().plot(kind = 'bar')
plt.xlabel('Education')
plt.ylabel('Count')
plt.title('Education level')
plt.show()


# In[21]:


#checking sex

sns.countplot(x = data['sex'], data = data)
plt.xticks(rotation = 90)
plt.show()


# In[22]:


#checking occupation

data['occupation'].value_counts().plot(kind = 'bar')
plt.show()


# In[23]:


#Checking race

sns.countplot(x = data['race'], data = data)
plt.xticks(rotation = 90)
plt.show()


# In[24]:


#checking marital-status

sns.countplot(x = data['marital-status'], data = data)
plt.xticks(rotation = 90)
plt.show()


# In[25]:


#checking relationship

sns.countplot(x = data['relationship'], data = data)
plt.xticks(rotation = 90)
plt.show()


# In[26]:


#checking native-country

data['native-country'].value_counts().plot(kind = 'bar')
plt.yticks(np.arange(5000,33000,3000))
plt.show()


# In[27]:


data.plot(kind='box',subplots=True,layout=(2,3),figsize=(14,14));


# From the figure it is clear that age and hours-per-week have outliers. So we should remove it

# In[28]:


#Outlier handling of age

Q1=np.percentile(data['age'],25,interpolation='midpoint')
Q2=np.percentile(data['age'],50,interpolation='midpoint')
Q3=np.percentile(data['age'],75,interpolation='midpoint')
IQR=Q3-Q1
print(Q1,Q2,Q3)
low_limit=Q1-1.5*IQR
up_limit=Q3+1.5*IQR
print('low_limit=',low_limit)
print('up_limit=',up_limit)
outlier=[]
for x in data['age']:
  if((x>up_limit) or (x<low_limit)):
    outlier.append(x)

indA=data['age']>up_limit
indA1=data.loc[indA].index
indB=data['age']<low_limit
indB1=data.loc[indB].index

data.drop(indA1,inplace=True)
data.drop(indB1,inplace=True)


# In[29]:


plt.boxplot(data['age'])
plt.title('Box plot of age after removal of outlier')


# In[30]:


#Outlier handling of hours-per-week

Q1=np.percentile(data['hours-per-week'],25,interpolation='midpoint')
Q2=np.percentile(data['hours-per-week'],50,interpolation='midpoint')
Q3=np.percentile(data['hours-per-week'],75,interpolation='midpoint')
IQR=Q3-Q1
print(Q1,Q2,Q3)
low_limit=Q1-1.5*IQR
up_limit=Q3+1.5*IQR
print('low_limit=',low_limit)
print('up_limit=',up_limit)
outlier=[]
for x in data['hours-per-week']:
  if((x>up_limit) or (x<low_limit)):
    outlier.append(x)

indA=data['hours-per-week']>up_limit
indA1=data.loc[indA].index
indB=data['hours-per-week']<low_limit
indB1=data.loc[indB].index

data.drop(indA1,inplace=True)
data.drop(indB1,inplace=True)


# In[31]:


plt.boxplot(data['hours-per-week'])
plt.title('Box plot of hours-per-week after removal of outlier')


# In[32]:


data.shape


#  Outliers are handled and shape reduced from (32561,14) to (23499,11)

# In[33]:


# Importing LabelEncoder
from sklearn.preprocessing import LabelEncoder
label= LabelEncoder()


# In[34]:


data['workclass']=label.fit_transform(data['workclass'])
data['education']=label.fit_transform(data['education'])
data['occupation']=label.fit_transform(data['occupation'])
data['sex']=label.fit_transform(data['sex'])
data['salary']=label.fit_transform(data['salary'])
data['race']=label.fit_transform(data['race'])
data['native-country']=label.fit_transform(data['native-country'])
data['marital-status']=label.fit_transform(data['marital-status'])
data['relationship']=label.fit_transform(data['relationship'])


# In[35]:


data

