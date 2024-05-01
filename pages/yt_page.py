import streamlit as st
from uliti import yt_predict,video_id
import plotly.express as px 


with st.container():
    link=st.text_input("Paste video link here")
    if st.button('Predict'):
        video_Id=video_id(link)
        prediction = yt_predict(video_Id)
        st.dataframe(prediction)
   
    
    if st.button('chart'):
        video_Id=video_id(link)
        prediction = yt_predict(video_Id)
        pie_chart=px.pie(prediction,title='$$$$$',values='sentiment')
        st.plotly_chart(pie_chart)    

    
    
        

