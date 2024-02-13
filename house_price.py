#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd


# In[29]:


import numpy as np


# In[50]:


import plotly.express as px


# In[30]:


data = pd.read_csv(r"C:/Users/AKUM1KOR/Desktop/KAGGLE/train.csv")


# In[31]:


data.head()


# In[32]:


data.shape


# In[33]:


data.info()


# In[34]:


import seaborn as sns


# In[35]:


import matplotlib.pyplot as plt


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


data.describe()


# In[38]:


data.columns


# In[39]:


data.isna().sum()


# In[40]:


data["ADDRESS"]= data["ADDRESS"].str.split(",").str[-1]


# In[41]:


data.head()


# In[42]:


def map_city(city):
    if city in ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai', 'Pune']:
        return 'tier1'
    elif city in ['Agra', 'Ajmer', 'Aligarh', 'Amravati', 'Amritsar', 'Asansol', 'Aurangabad', 'Bareilly', 
                  'Belgaum', 'Bhavnagar', 'Bhiwandi', 'Bhopal', 'Bhubaneswar', 'Bikaner', 'Bilaspur', 'Bokaro Steel City', 
                  'Chandigarh', 'Coimbatore', 'Cuttack', 'Dehradun', 'Dhanbad', 'Bhilai', 'Durgapur', 'Dindigul', 'Erode', 
                  'Faridabad', 'Firozabad', 'Ghaziabad', 'Gorakhpur', 'Gulbarga', 'Guntur', 'Gwalior', 'Gurgaon', 'Guwahati', 
                  'Hamirpur', 'Hubli–Dharwad', 'Indore', 'Jabalpur', 'Jaipur', 'Jalandhar', 'Jammu', 'Jamnagar', 'Jamshedpur', 
                  'Jhansi', 'Jodhpur', 'Kakinada', 'Kannur', 'Kanpur', 'Karnal', 'Kochi', 'Kolhapur', 'Kollam', 'Kozhikode', 
                  'Kurnool', 'Ludhiana', 'Lucknow', 'Madurai', 'Malappuram', 'Mathura', 'Mangalore', 'Meerut', 'Moradabad', 
                  'Mysore', 'Nagpur', 'Nanded', 'Nashik', 'Nellore', 'Noida', 'Patna', 'Pondicherry', 'Purulia', 'Prayagraj', 
                  'Raipur', 'Rajkot', 'Rajahmundry', 'Ranchi', 'Rourkela', 'Ratlam', 'Salem', 'Sangli', 'Shimla', 'Siliguri', 
                  'Solapur', 'Srinagar', 'Surat', 'Thanjavur', 'Thiruvananthapuram', 'Thrissur', 'Tiruchirappalli', 'Tirunelveli', 
                  'Tiruvannamalai', 'Ujjain', 'Bijapur', 'Vadodara', 'Varanasi', 'Vasai-Virar City', 'Vijayawada', 'Visakhapatnam', 
                  'Vellore', 'Warangal']:
        return 'tier2'
    else:
        return 'tier3'
    
data['city_tier'] = data['ADDRESS'].apply(map_city)


# In[43]:


print(data["BHK_OR_RK"].value_counts())


# In[44]:


plt.bar(["BHK","RK"],data["BHK_OR_RK"].value_counts())


# In[45]:


data.drop(['POSTED_BY',"BHK_OR_RK","ADDRESS"], axis=1, inplace = True)
data.head()


# In[21]:





# In[46]:


data["LONGITUDE"],data["LATITUDE"]=data["LATITUDE"],data["LONGITUDE"]


# In[47]:


data=data[data["LONGITUDE"].between(65,96) & data["LATITUDE"].between(7,36)]


# In[48]:


data


# In[61]:


fig = px.density_mapbox(data,  lon = 'LONGITUDE',lat = 'LATITUDE',
                        radius = 8,
                        zoom = 6,
                        mapbox_style = 'open-street-map')
fig.show()


# In[60]:


get_ipython().system('pip install plotly')


# In[62]:


import plotly.express as px


# In[63]:


fig = px.density_mapbox(data,  lon = 'LONGITUDE',lat = 'LATITUDE',
                        radius = 8,
                        zoom = 6,
                        mapbox_style = 'open-street-map')
fig.show()


# In[64]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score


# In[65]:


encoder = LabelEncoder()
data["city_tier"]= encoder.fit_transform(data["city_tier"])


# In[66]:


training_data = data.drop(["TARGET(PRICE_IN_LACS)"],axis=1)
target = data["TARGET(PRICE_IN_LACS)"]


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.2, random_state=42)


# In[68]:


scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# In[69]:


model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_train, y_train)


# In[70]:


model = DecisionTreeRegressor()
model.fit(X_train,y_train)
preds=model.predict(X_test)
print(r2_score(preds,y_test))


# In[71]:


model = RandomForestRegressor()
model.fit(X_train,y_train)
preds=model.predict(X_test)
print(r2_score(preds,y_test))


# In[ ]:




