import streamlit as st
from uliti import predict

st.markdown('#**PREDICT SENTMENT** ')
st.divider()
inputext=st.text_input('Enter the sentance :')
if st.button('proceed'):
    prediction=predict(inputext)
    if prediction==1:
        st.write(prediction,"😡")
    if prediction==2:
        st.write(prediction,"😑")
    if prediction==3:
        st.write(prediction,"🙄")
    if prediction==4:
        st.write(prediction,"😊")
    if prediction==5:
        st.write(prediction,"😁")
