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


# ***Standardization***

# In[36]:


# Setting the feature and targert variables
X=data.drop(columns=['salary'],axis=1)
y=data['salary']
X.head()


# In[37]:


# Satandardizing the feature using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[38]:


# Splitting the data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.30, random_state = 42)


# In[39]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Modeling

# In[40]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,precision_score,recall_score,classification_report


# ### a. Logistic Regression

# In[41]:


# Fitting the training data to Logostic Regression Model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=2000)
lr.fit(X_train,y_train)


# In[42]:


# Predicting using test data
pred1=lr.predict(X_test)
pred1


# In[43]:


# Checking the confusion matrix and accuracy of the model
con_lr=confusion_matrix(y_test,pred1)
print("The confusion matrix of logistic regression is \n",con_lr)

ac_lr=accuracy_score(y_test,pred1)
print('Accuracy:',ac_lr*100)


# In[44]:


# Classification Report
print(classification_report(y_test, pred1))


# ### b . K Nearest Neighbour Classifier

# In[45]:


# Fitting the training data to KNNClassifier Model
from sklearn.neighbors import KNeighborsClassifier
acc_values=[]
neighbors=np.arange(70,90)
for k in neighbors:
    knn=KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    knn.fit(X_train, y_train)
    pred2=knn.predict(X_test)
    ac_knn=accuracy_score(y_test,pred2)
    acc_values.append(ac_knn)


# In[46]:


acc_values


# In[47]:


# Checking the average accuracy of the model
avg_acc = np.array(acc_values).mean()
print('Accuracy: ', avg_acc * 100)


# In[48]:


plt.plot(neighbors,acc_values,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')


# In[49]:


# Classification Report
print(classification_report(y_test, pred2))


# ### c. Support Vector Machine

# In[50]:


# Fitting the training data to SVC Model
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)


# In[51]:


# Predicting using test data
pred3=svc.predict(X_test)
pred3


# In[52]:


# Checking the confusion matrix and accuracy of the model
con_svc=confusion_matrix(y_test,pred3)
print("The confusion matrix of decision tree is \n",con_svc)

ac_svc=accuracy_score(y_test,pred3)
print('Accuracy:',ac_svc*100)


# In[53]:


# Classification Report
print(classification_report(y_test, pred3))


# ### d. Decision Tree classifier

# In[54]:


# Fitting the training data to Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)


# In[55]:


# Predicting using test data
pred4=dt_clf.predict(X_test)
pred4


# In[56]:


# Checking the confusion matrix and accuracy of the model
con_dtr=confusion_matrix(y_test,pred4)
print("The confusion matrix of decision tree is \n",con_dtr)

ac_dt=accuracy_score(y_test,pred4)
print('Accuracy:',ac_dt*100)


# In[57]:


# Classification Report
print(classification_report(y_test, pred4))


# ### e.Random Forest Classifier

# In[58]:


# Fitting the training data to Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)


# In[59]:


# Predicting using test data
pred5=rf.predict(X_test)
pred5


# In[60]:


# Checking the confusion matrix and accuracy of the model
con_rf=confusion_matrix(y_test,pred5)
print("The confusion matrix of random forest is \n",con_rf)

ac_rf=accuracy_score(y_test,pred5)
print('Accuracy:',ac_rf*100)


# In[61]:


# Classification Report
print(classification_report(y_test, pred5))


# #### Summary of accuracies of different models

# In[62]:


# Tabulating the accuracies of different models
from tabulate import tabulate
table = [['Model', 'Accuracy'], ['Logistic Regression',ac_lr], ['KNN',avg_acc], ['Decision tree',ac_dt], ["SVM",ac_svc], ['Random Forest',ac_rf]]
print(tabulate(table,headers='firstrow',tablefmt='fancy_grid'))


# From the table, it is clear that Random forest have better accuracy compared to others.So Random forest is taken as our model to predict the salary.So we can tune this to check whether the performance is improving.
# 
# **Hyper parameter tuning of Random Forest Model**
# 
# 

# In[63]:


# Fitting the training set
rf=RandomForestClassifier(criterion='gini',max_depth=10,n_estimators=600)
rf.fit(X_train,y_train)


# In[64]:


# Passing the test set
y_pr=rf.predict(X_test)
y_pr


# In[65]:


# Checking the accuracy
acc_sc=accuracy_score(y_test,y_pr)*100
print('Accuracy: ', acc_sc)


# Hyper parameter tuning improved the accuracy of Random forest modeling to 82.61%.
# 
# So we can take random forest classifier to build our model.

# In[ ]:




