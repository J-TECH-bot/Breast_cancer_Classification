# 🩺 Breast Cancer Classification

A Machine Learning + Deep Learning powered web application for breast cancer detection based on medical data. This project uses trained models to classify whether a tumor is Malignant (cancerous) or Benign (non-cancerous).

The app is deployed using Streamlit for an easy-to-use interface, where users can input relevant medical features and get predictions instantly.

📌 Features

✅ Predict Malignant or Benign tumor from input features.

✅ Built with Deep Learning models (L_Model.h5 and NN_Model.h5).

✅ User-friendly Streamlit web app interface.

✅ Supports instant prediction without complex installations.

✅ Trained on a Breast Cancer Dataset (data.csv).


📊 Dataset

The dataset (data.csv) contains breast cancer diagnostic data with features like:

Radius Mean

Texture Mean

Perimeter Mean

Area Mean

Smoothness Mean

… and other computed medical features

Target variable:

0 → Benign

1 → Malignant

Dataset Source: Breast Cancer Wisconsin (Diagnostic) Dataset

🧠 Model Details

Two deep learning models were trained:

L_Model.h5 → A larger deep learning architecture for higher accuracy.

NN_Model.h5 → A simpler neural network for faster predictions.

Both models use features from the dataset to classify cancer type.

🚀 Installation & Usage
1️⃣ Clone the repository
git clone https://github.com/J-TECH-bot/Breast_cancer_Classification.git
cd Breast_cancer_Classification

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app locally
streamlit run app.py

🌐 Deployment

This project is deployed using Streamlit Cloud.
You can try the live app here: 🔗 Live Demo:- https://breastcancerclassifier0.streamlit.app/ 
