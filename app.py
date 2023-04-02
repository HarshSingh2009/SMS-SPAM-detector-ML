from MLPipeline import MlPipeline
import streamlit as st  

# App Building

st.title('SMS spam prediction using ML')
st.subheader('Enter you SMS message to know if its spam or ham: ')

with st.sidebar:
    st.subheader('Chose your ML model: ')
    choices = st.selectbox('Select your model', ['SVM Model', 'Naive Byes Model'])

with st.form(key='form1'):
    sms_message = st.text_area('Enter SMS/Email Text Below : ', height=300) 
    analyse_button = st.form_submit_button(label='Analyse Text')
    if analyse_button:
        ml_object = MlPipeline(message=sms_message) 
        message = ml_object.transform_text()
        predict = ml_object.ml_pipeline_predict(choices=choices)
        if predict[0] == 1:
            st.success(f'{choices} : It is a SPAM !!!')
        else:
            st.text(f'{choices} : It is not a SPAM !!!')