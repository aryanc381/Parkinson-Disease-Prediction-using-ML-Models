#!/usr/bin/env python
# coding: utf-8

# # **Parkinson Disease Prediction**

# ## 1. Importing data and libraries

# In[83]:


import numpy as np
import pandas as pd


# In[85]:


data = pd.read_csv('C:\\Users\\conta\\Desktop\\parkinson\\parkinsondisease.csv')


# In[86]:


data.head()


# ## 2. Feature Extraction and Selection
# 
# 1. **MDVP:Fo(Hz):** The average fundamental frequency measured in Hz.
# 2. **MDVP:Fhi(Hz):** The highest fundamental frequency measured in Hz.
# 3. **MDVP: Flo(Hz):** The lowest fundamental frequency measured in Hz.
# 4. **MDVP:Jitter(%):** The percentage of the absolute jitter, which is defined as the variation in the fundamental frequency, relative to the fundamental frequency.
# 5. **MDVP:Jitter(Abs):** The absolute jitter, measured in Hz.
# 6. **MDVP:RAP:** The relative amplitude perturbation, which is defined as the average absolute difference between consecutive signal periods divided by the average signal period.
# 7. **MDVP:PPQ:** The pitch period perturbation quotient, which is similar to the RAP but only takes into account the pitch periods.
# 8. **Jitter:** DDP: The average absolute difference between consecutive signal periods divided by three times the average signal period.
# 9. **MDVP:Shimmer:** The amplitude variation of the signal in dB.
# 10. **MDVP:Shimmer(dB):** The amplitude variation of the signal in dB.
# 11. **Shimmer:APQ3:** The amplitude perturbation quotient measured in dB, which only takes into account the first three harmonics.
# 12. **Shimmer:APQ5:** The amplitude perturbation quotient measured in dB, which only takes into account the first five harmonics.
# 13. **MDVP:APQ:** The amplitude perturbation quotient measured in dB, which takes into account all harmonics.
# 14. **Shimmer:DDA:** The average absolute difference between consecutive signal periods divided by the signal length, measured in dB.
# 15. **NHR:** The ratio of the noise to the harmonic components in the voice.
# 16. **HNR:** The ratio of the energy in the harmonic components to the energy in the noise.
# status: A binary variable that indicates whether the patient has Parkinson's disease (1) or not (0).
# 17. **RPDE:** The relative average perturbation entropy, which is a measure of the unpredictability of the signal.
# 18. **DFA:** The detrended fluctuation analysis, which measures the scaling behavior of the signal.
# 19. **spread1:** A nonlinear measure of the speech signal.
# 20. **spread2:** Another nonlinear measure of the speech signal.
# 21. **D2:** The correlation dimension, which measures the fractal dimension of the signal.
# 22. **PPE:** The pitch period entropy, which measures the disorder and complexity of the signal.
# 

# ## Data Verification
# 
# 1. **Dimensionality** of the data.
# 2. **Null Values** if present in data have to be checked.
# 3. **Duplicate Values** if present need to be removed.
# 4. **Data Filtering** needs to be done for the above two steps.
# 5. **Getting info** of the data
# 6. **Statistical measure** of the data should be seen for a better overview.

# In[87]:


data.shape


# In[88]:


data.isnull().sum()


# In[89]:


data.duplicated().sum()


# In[90]:


data.info()


# In[91]:


data.describe()


# In[92]:


data['status'].value_counts()

# Parkinson +ve [1] : 147
# Parkinson -ve [0] : 48


# In[93]:


data.groupby('status').mean()


# ## **3. Splitting dataset into train and test**

# In[94]:


x = data.drop(['status', 'name'], axis=1)
y = data['status']


# In[95]:


x.shape


# In[96]:


y.shape


# In[97]:


from sklearn.model_selection import train_test_split


# In[98]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# ## **4. Standardization**

# In[99]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()


# In[105]:


scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)


# In[106]:


x_train


# ## **5. Model Prediction**

# In[107]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[108]:


svc = SVC()
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
accuracy_score(y_test, y_predict)


# In[109]:


models = {
    "lg" : LogisticRegression(),
    "knc" : KNeighborsClassifier(),
    "svc" : SVC(),
    "gnb" : GaussianNB(),
    "dtc" : DecisionTreeClassifier(),
    "rfc" : RandomForestClassifier()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    
    print(f"{name} with accuracy : ", accuracy_score(y_test, y_predict))


# In[ ]:




