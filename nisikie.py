import pandas as pd
import streamlit as st
import joblib

loaded_model = joblib.load('random_forest_model.pkl')

# Defining the function which will make the prediction using the data which the user inputs
def prediction(GENDER, MARITAL_STATUS, LOST_WEIGHT, TWO_WEEK_SAD, TROUBLE_SLEEPING, TAKE_DRUGS, SUICIDE_THOUGHT,
               FAILURE_DECISIONS, DEBT, DEPRESSED):
    # Pre-processing user input
    if GENDER == "Male":
        GENDER = 0
    else:
        GENDER = 1

    if MARITAL_STATUS == "Unmarried":
        MARITAL_STATUS = 0
    else:
        MARITAL_STATUS = 1

    if DEBT == "Unclear Debts":
        DEBT = 0
    else:
        DEBT = 1

    if LOST_WEIGHT == "No":
        LOST_WEIGHT = 0
    else:
        LOST_WEIGHT = 1

    if TWO_WEEK_SAD == "No":
        TWO_WEEK_SAD = 0
    else:
        TWO_WEEK_SAD = 1

    if TROUBLE_SLEEPING == "No":
        TROUBLE_SLEEPING = 0
    else:
        TROUBLE_SLEEPING = 1

    if TAKE_DRUGS == "No":
        TAKE_DRUGS = 0
    else:
        TAKE_DRUGS = 1

    if SUICIDE_THOUGHT == "No":
        SUICIDE_THOUGHT = 0
    else:
        SUICIDE_THOUGHT = 1

    if FAILURE_DECISIONS == "No":
        FAILURE_DECISIONS = 0
    else:
        FAILURE_DECISIONS = 1

    if DEPRESSED == "No":
        DEPRESSED = 0
    else:
        DEPRESSED = 1

    # Making predictions using only the 10 selected features
    prediction = loaded_model.predict([[GENDER, MARITAL_STATUS, LOST_WEIGHT, TWO_WEEK_SAD, TROUBLE_SLEEPING,
                                        TAKE_DRUGS, SUICIDE_THOUGHT, FAILURE_DECISIONS, DEBT, DEPRESSED]])

    if prediction == 0:
        pred = 'Not Depressed'
    else:
        pred = 'Depressed'
    return pred


def main():
# Front-end elements of the web page
# Add a title and a description
    st.title('Nisikie (Depression Prediction)')
    st.markdown('Answer the following questions to generate a prediction.')

# Create a sidebar for additional information or instructions
st.sidebar.title('Nisikie')
st.sidebar.markdown('### ℹ️ Welcome to Nisikie')
st.sidebar.markdown("🌟 The Depression Prediction Application: **Nisikie** translates to 'Hear Me' in Swahili. We aim for this application to be a valuable resource in saving lives.")

st.sidebar.markdown('### Instructions')
st.sidebar.markdown('👉 Please select your answers to the questions on the left and click **Generate** to get the prediction result.')
st.sidebar.markdown('⚠️ **Disclaimer:** The prediction provided by this application should not be considered as a substitute for professional medical advice.')


# Following lines create boxes in which user can enter data required to make prediction
with st.form('prediction_form'):
    st.header('Nisikie')
    col1, col2 = st.columns(2)
    with col1:
        GENDER = st.selectbox('Gender', ("Male", "Female"))
        MARITAL_STATUS = st.selectbox('Marital Status', ("Unmarried", "Married"))
        LOST_WEIGHT = st.selectbox('Lost weight past two weeks?', ("No", "Yes"))
        TWO_WEEK_SAD = st.selectbox('Have you been eating more than usual?', ("No", "Yes"))
    with col2:
        TROUBLE_SLEEPING = st.selectbox('Having trouble sleeping?', ("Yes", "No"))
        TAKE_DRUGS = st.selectbox('Have you been drinking a lot or taking drugs?', ("No", "Yes"))
        SUICIDE_THOUGHT = st.selectbox('Have you been having suicidal thoughts?', ("No", "Yes"))
        FAILURE_DECISIONS = st.selectbox('Do you feel like you are a failure?', ("No", "Yes"))
    DEBT = st.selectbox('Are you in debt?', ("No", "Yes"))
    DEPRESSED = st.text_input("Are you depressed?")
    submit_button = st.form_submit_button(label='Generate')

# When 'Generate' is clicked, make the prediction and display the result
if submit_button:
    result = prediction(GENDER, MARITAL_STATUS, LOST_WEIGHT, TWO_WEEK_SAD, TROUBLE_SLEEPING,
                        TAKE_DRUGS, SUICIDE_THOUGHT, FAILURE_DECISIONS, DEBT, DEPRESSED)
    st.markdown('---')
    st.header('Prediction Result')
    if result == 'Approved':
        st.success('Based on the provided information, you are predicted to be **{}**'.format(result))
    else:
        st.error('Based on the provided information, you are predicted to be **{}**'.format(result))
