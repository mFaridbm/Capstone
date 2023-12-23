import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler as scaler

# Load the trained model
model_filename = 'capstone_employment.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)
    
# Load the scaler from the file
with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

def get_attributes_in_list(model):
    return [col for col in model.feature_names_in_]
 
# Function to preprocess input data
def preprocess_input_data(input_data):
    # Assuming input_data is a dictionary containing user input
    features = pd.DataFrame(input_data, index=[0])

    # Convert categorical features to dummy variables
    categorical_features = ['Job_level', 'Nature_of_Scope', 'education']
    
    # Remove drop_first. 
    df_dummies = pd.get_dummies(features[categorical_features])
    features = pd.concat([df_dummies, features[['years_of_experience']]], axis=1)
    
    # Standardize the numerical feature
    features['years_of_experience'] = loaded_scaler.transform(features[['years_of_experience']])

    return features

# Streamlit app
st.title('Salary Prediction App')

# User input form
job_level = st.selectbox('Job Level', ['Executive', 'Middle Management', 'Professional', 'Fresh/entry level', 'Manager', 'Non-executive', 'Senior Management', 'Senior Executive', 'Junior Executive'])
nature_of_scope = st.selectbox('Nature of Scope', ['engineer', 'scientist', 'lead', 'analyst', 'manager', 'developer', 'sales', 'consultant', 'specialist', 'General', 'administrator', 'research', 'operations', 'management', 'accounts', 'associate', 'strategist', 'designer', 'assistant', 'executive', 'technician', 'architect', 'tester', 'administration', 'cloud'])
education = st.selectbox('Education', ['degree', 'No Education Required', 'diploma', 'phd', 'nitec', 'o levels', 'masters'])
years_of_experience = st.slider('Years of Experience', min_value=0, max_value=20, value=5)

# Make predictions when the user clicks the "Predict" button
if st.button('Predict'):
    # Create input data dictionary
    input_data = {
        'Job_level': job_level,
        'Nature_of_Scope': nature_of_scope,
        'education': education,
        'years_of_experience': years_of_experience
    }

    # Preprocess input data
    input_features = preprocess_input_data(input_data)
    
    # Sync features based on model. 
    # Those that are not present in input_features are put as zero.
    feature_list = get_attributes_in_list(model)
    for col in feature_list:
        if col not in input_features.columns:
            input_features[col] = 0

    # To align the features in the same order as model
    input_features = input_features[feature_list]
    
    # After loading the scaler
    # Make prediction
    prediction = model.predict(input_features)

    # Display the prediction
    st.success(f'Predicted Salary: ${prediction[0]:,.2f}')