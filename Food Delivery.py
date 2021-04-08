#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


zomato_orgnl = pd.read_csv(r"D:\Dr. Kumar\zomato.csv")


# In[3]:


zomato_orgnl.head()


# In[4]:


zomato_orgnl.isnull().sum()


# In[5]:


zomato_orgnl.info()


# In[6]:


zomato = zomato_orgnl.drop(['url', 'dish_liked', 'phone'], axis = 1)
zomato.columns


# In[7]:


zomato.rename({"approx_cost(for two people)" : 'approx_cost_for_2_people',
              'listed_in(type)' : 'listed_in_type',
              'listed_in(city)' : "listed_in_city"}, axis = 1, inplace = True)
zomato.columns


# In[8]:


remove_comma = lambda x : int(x.replace(",", '')) if type(x) == np.str and x != np.nan else x
zomato.votes = zomato.votes.astype("int")
zomato["approx_cost_for_2_people"] = zomato["approx_cost_for_2_people"].apply(remove_comma)


# In[9]:


zomato.info()


# In[10]:


zomato['rate'].unique()


# In[11]:


zomato = zomato.loc[zomato.rate != 'NEW']
zomato = zomato.loc[zomato.rate != '-'].reset_index(drop = True)


# In[12]:


zomato['rate'].unique()


# In[13]:


remove_slash = lambda x : x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')


# In[14]:


zomato['rate'].unique()


# In[15]:


zomato.info()


# In[16]:


def encode(zomato):
    for col in zomato.columns[~zomato.columns.isin(['rate', 'approx_cost_for_2_people', 'votes'])]:
        zomato[col] = zomato[col].factorize()[0]
    return zomato

zomato_en = encode(zomato.copy())


# In[17]:


zomato_en.info()


# In[18]:


zomato_en["rate"] = zomato_en['rate'].fillna(zomato_en['rate'].mean())
zomato_en["approx_cost_for_2_people"] = zomato_en['approx_cost_for_2_people'].fillna(zomato_en['approx_cost_for_2_people'].mean())


# In[19]:


zomato_en.isna().sum()


# In[20]:


corr = zomato_en.corr(method = 'kendall')
corr


# In[21]:


plt.figure(figsize = (15, 8))
sb.heatmap(corr, annot = True)


# In[22]:


from sklearn.model_selection import train_test_split
x = zomato_en.iloc[:, [2, 3, 5, 6, 7, 8, 9, 11]]
y = zomato_en['rate']


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 353)


# In[24]:


x_train.head()


# In[25]:


y_train.head()


# In[26]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train, y_train)


# In[27]:


y_pred = reg.predict(x_test)


# In[28]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[29]:


from sklearn.tree import DecisionTreeRegressor

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 21)


# In[30]:


dtree = DecisionTreeRegressor(min_samples_leaf=0.0001)


# In[31]:


dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)


# In[32]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[33]:


from sklearn.ensemble import RandomForestRegressor

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 173)


# In[34]:


rforest = RandomForestRegressor()


# In[35]:


rforest.fit(x_train, y_train)
y_pred_forest = rforest.predict(x_test)


# In[36]:


r2_score(y_test, y_pred_forest)


# Random Forest Regressor has an r2 score of over .90 therefore Neural Network not required

# In[37]:


fig = plt.figure(figsize = (22, 7))
loc = sb.countplot(x='location', data = zomato_orgnl, palette = "Set1")
loc.set_xticklabels(loc.get_xticklabels(), rotation = 90, ha = "right")

plt.ylabel("frequency", size = 15)
plt.xlabel("location", size = 18)
plt.title("Num of restaurants in a location", size = 22, pad = 20)


# In[38]:


fig = plt.figure(figsize=(18, 5))
rest = sb.countplot(x = "rest_type", data = zomato_orgnl, palette="Set1")
rest.set_xticklabels(rest.get_xticklabels(), rotation = 90, ha = "right")


plt.ylabel("frequency", size = 16)
plt.xlabel("Restaurant Type", size = 16)
plt.title("Num of restaurants by type", size = 22, pad = 20)


# In[40]:


plt.figure(figsize=(16, 7))
chains = zomato_orgnl["name"].value_counts()[:20]
sb.barplot(x = chains, y = chains.index, palette = "Set1")

plt.title("Most famous restaurant chain in Bangalore", size = 22, pad = 20)
plt.xlabel("Number of outlets", size = 16)


# In[42]:


plt.figure(figsize=(16, 7))
zomato_orgnl['online_order'].value_counts().plot.bar()
plt.title('Online orders', fontsize = 22)
plt.ylabel("Frequency", size = 16)


# In[45]:


fig, ax = plt.subplots(figsize = (15, 7))
sb.distplot(zomato_en['approx_cost_for_2_people'], color = "magenta")
ax.set_title("Approx Cost for two people distrubution", size = 22, pad = 16)

plt.xlabel("Approx cost for two people", size = 16)
plt.ylabel("Percentage of restaurants", size = 16)


# In[46]:


plt.figure(figsize = (16, 8))
cuisines = zomato_orgnl["cuisines"].value_counts()[:15]
sb.barplot(cuisines, cuisines.index)

plt.title("Most popular cuisines of Bangalore", size = 22, pad = 16)
plt.xlabel("Number of restaurants", size=16)


# In[ ]:




