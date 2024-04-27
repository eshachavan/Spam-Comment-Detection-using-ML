# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:47:42 2024

@author: Esha
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("C:/Users/Esha/Desktop/Project_ML/model (2).pkl", "rb"))
cv = pickle.load(open("C:/Users/Esha/Desktop/Project_ML/cv.pkl", "rb"))

st.title("Spam Comment Detection")
comment = st.text_input("Comment")

if st.button("Detect"):
	test = cv.transform([comment]).toarray()
	res = model.predict(test)
	print(res)
	st.success("Detected: " + str(res[0]))