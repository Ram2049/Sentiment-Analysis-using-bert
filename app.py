import streamlit as st
import requests
import json
from uliti import predict
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import requests
import re 

st.markdown('# SENTIMENT ANALYSIS USING :rainbow[BERT] ')

st.divider()

with st.container():
    r_col,l_col=st.columns([0.5,0.5])
    with r_col:
        st.markdown("### *MAKE SENTIMENT ANALYSIS ON A SENTENCE*")
        if r_col.button("Predict"):
            st.switch_page("pages/sentence.py")
    with l_col :
         st.markdown("### *MAKE SENTIMENT ANALYSIS ON A YOUTUBE COMMENTS*")
         if l_col.button("predict"):
             st.switch_page('pages/yt_page.py')
         
