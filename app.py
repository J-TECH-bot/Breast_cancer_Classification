import numpy as np
import streamlit as st
import pandas as pd
#from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the saved model with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('NN_Model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define the features for the prediction (fixed spacing issues)
features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean', 
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'fractal_dimension_mean',
    'radius_se',
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
    'texture_worst',
    'perimeter_worst',
    'area_worst',
    'smoothness_worst',
    'compactness_worst',
    'concavity_worst',
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

# Check if model is loaded
if model is None:
    st.error("Model could not be loaded. Please check if NN_Model.h5 file is present.")
    st.stop()

# Create input fields for each feature
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(feature, value=0.0, step=0.01)

# Create a button to predict the cancer risk
if st.button("Predict Cancer"):
    try:
        # Create a pandas DataFrame from the user input
        df = pd.DataFrame([user_input])
        input_data_as_numpy_array = np.asarray(df)

        # reshape the numpy array as we are predicting for one data point
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Preprocess the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(input_data_reshaped)
        
        # Predict the cancer risk using the loaded model
        prediction = model.predict(scaled_data, verbose=0)[0][0]

        # Display the prediction
        st.write(f"**Predicted Cancer Risk:** {prediction:.4f}")
        
        # Add interpretation
        if prediction > 0.5:
            st.warning("⚠️ **High Risk**: The prediction suggests a higher likelihood of malignancy.")
        else:
            st.success("✅ **Low Risk**: The prediction suggests a lower likelihood of malignancy.")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please check your input values and try again.")
