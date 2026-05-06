#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib


# In[13]:


kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")


# In[14]:


st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment.")


# In[15]:


age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)


# In[16]:


input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits]
})

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")


# In[ ]:





# In[ ]:




