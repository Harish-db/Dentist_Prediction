#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc


# In[7]:


st.title('Gender Predictions Using Dental Mesurements')

with open ('std_top6_df.pkl','rb')as file:
    std_df=pickle.load(file)
    
with open ('rfc.pkl','rb') as file:
    rfc_model=pickle.load(file)
    
st.sidebar.header('Input Parameters')

left_canine_width_casts=st.number_input('left canine width casts(mm)')
left_canine_width_intraoral=st.number_input('left canine width intraoral(mm)')
right_canine_width_casts=st.number_input('right canine width casts(mm)')
right_canine_width_intraoral=st.number_input('right canine width intraoral(mm)')
intercanine_distance_casts=st.number_input('intercanine distance casts(mm)')
right_canine_index_casts=st.number_input('right canine index casts(mm)')

if st.button('Predict Gender'):
    input_data=[left_canine_width_casts,
               left_canine_width_intraoral,
               right_canine_width_casts,
               right_canine_width_intraoral,
               intercanine_distance_casts,
               right_canine_index_casts]
    column_names=['left canine width casts',
                 'left canine width intraoral',
                 'right canine width casts',
                 'right canine width intraoral',
                 'intercanine distance casts',
                 'right canine index casts']
    
    input_df=pd.DataFrame([input_data], columns=column_names)
    gender_pred=rfc_model.predict(input_df)
    
    if gender_pred[0]==0:
        st.write('The Predicted Gender is Female')
    else:
        st.write('The Predicted Gender is Male')
        

