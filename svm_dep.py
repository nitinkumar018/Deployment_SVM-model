#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
pickle_in=open('support.pkl','rb')
svc_model=pickle.load(pickle_in)


# In[2]:


# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.svm import SVC

st.title('Model Deployment: Support Vector Machine')

st.sidebar.header('User Input Parameters')

def user_input_features():
    stalk_height = st.sidebar.number_input("Insert the height")
    cap_diameter = st.sidebar.number_input("Insert the diameter")
    data1 = {'stalk_height':stalk_height,
             'cap_diameter':cap_diameter}
    features = pd.DataFrame(data1,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv(r"C:\Users\nitin\Downloads\Latest DS material\SVM - Streamlit\mushroom.csv")
data.drop(["Unnamed: 0"],inplace=True,axis = 1)
data = data.dropna()

X = data.values[:,22:24]
Y = data.values[:,-1]
svc_model= SVC(kernel='rbf', gamma=0.01, C =20)
svc_model.fit(X,Y)

Y_pred = svc_model.predict(df)
st.write('poisonous' if Y_pred ==1.0 else 'edible')


# In[ ]:




