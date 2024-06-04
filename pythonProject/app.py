import numpy as np
import streamlit as st
import pandas as pd
#from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = tf.keras.models.load_model('NN_Model.h5')  # Replace 'your_model.h5' with your model file

# Define the features for the prediction
features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean ',
    'smoothness_mean', 'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'fractal_dimension_mean',
    'radius_se ',
    'texture_se',
    'perimeter_se',
    'area_se',
    'smoothness_se',
    'compactness_se',
    'concavity_se',
    'concave points_se',
    'symmetry_se',
    'fractal_dimension_se',
    'radius_worst',
    'texture_worst ',
    'perimeter_worst',
    'area_worst ',
    'smoothness_worst',
    'compactness_worst',
    'concavity_worst ',
    'concave points_worst',
    'symmetry_worst',
    'fractal_dimension_worst'
]

# Load the image
#image = Image.open('breast_cancer_image.jpg')  # Replace 'breast_cancer_image.jpg' with your image file

# Create the Streamlit page
st.title("Breast Cancer Detection")
#st.image(image, caption='Image for visualization', use_column_width=True)
st.header("Enter Cell Details")

# Create input fields for each feature
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(feature,)

# Create a button to predict the cancer risk
if st.button("Predict Cancer"):
    # Create a pandas DataFrame from the user input
    df = pd.DataFrame([user_input])
    input_data_as_numpy_array = np.asarray(df)

    # reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    # Preprocess the data
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(input_data_reshaped)
    # Predict the cancer risk using the loaded model
    prediction = model.predict(scaled_data)[0][0]

    # Display the prediction
    st.write(f"**Predicted Cancer Risk:** {prediction}")

    # You can add more information or logic based on the prediction here
