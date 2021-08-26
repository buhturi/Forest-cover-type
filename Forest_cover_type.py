#!/usr/bin/env python
# coding: utf-8

# In[168]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[143]:


data = pd.read_csv(r"C:\Users\20114\Desktop\Fatura_task\train.csv")
test_data = pd.read_csv(r"C:\Users\20114\Desktop\Fatura_task\test.csv")


# In[192]:


#Data cleaning
data.info()


# In[188]:


#check for null values and count it
print(data.isnull().sum())


# In[190]:


#get duplicates 
data.duplicated()


# In[146]:


print(data.shape)
print(test_data.shape)


# In[147]:


data.columns


# In[184]:


#visualize the relation between cover type and numeric features
l = ['Elevation','Aspect', 'Slope','Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
        'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
i=1
rows = 5
cols = 2
for col in l:
    ax = fig.add_subplot(rows, cols, i)
    sns.barplot(x="Cover_Type", y=col, data=data)
    plt.ylabel(col)
    i+=1


# In[180]:


#visualize the relation between cover type and categorical features
l = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
i=1
rows = 10
cols = 5
for col in l:
    ax = fig.add_subplot(rows, cols, i)
    sns.barplot(x="Cover_Type", y=col, data=data)
    plt.ylabel(col)
    i+=1


# In[72]:


labels = data['Cover_Type']
features =['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']


# In[199]:


x = data.loc[:,features]
y = labels
x_train, x_val, y_train, y_val = train_test_split( x.values, y.values, test_size=0.1, random_state=5 )


# In[204]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
pred = knn.predict(x_val)
accuracy_score(y_val, pred)


# In[98]:


RF = RandomForestClassifier(n_estimators=1000, random_state=5)
RF.fit(x_train, y_train)
pred = RF.predict(x_val)
accuracy_score(y_val,pred)


# In[139]:


DT = DecisionTreeClassifier(max_depth=10)
DT.fit(x_train, y_train)
pred = DT.predict(x_val)
accuracy_score(y_val,pred)


# In[101]:


x_test = test_data.loc[:,features]
y_test = RF.predict(x_test)
y_test


# In[105]:


#Generate test_data csv file with cover types
test_data['Cover_Type'] = list(y_test)
test_data.to_csv('test_data_with_cover_type.csv')


# Conclusion
# 
# 1 - No missing data
# 
# 2 - No duplicates in data
# 
# 3 - the cover type is affected by many variables
# 
# 3 - KNN clasifier is the best with 0.87 val accuracy

# In[ ]:




