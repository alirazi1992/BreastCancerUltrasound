import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Model Testing"])
if page == "Home":
    st.title("Welcome to Breast Cancer Detection")
    st.write("This web app is designed for breast cancer detection using deep learning models.")
    st.write("Please select 'Model Testing' from the sidebar to test the model.")
    
    # Add a summary of using machine learning in breast cancer detection
    st.subheader("Machine Learning in Breast Cancer Detection")
    st.write("Machine learning plays a vital role in breast cancer detection, helping medical professionals make more accurate diagnoses.")
    st.write("By analyzing medical images, such as ultrasound scans, machine learning models can assist in early detection and improve patient outcomes.")
    st.write("This web app uses a deep learning model to predict the likelihood of malignancy in ultrasound images.")
    st.write("Upload an ultrasound image to test the model.")
    
    # Add the video to the home page
    video_path = "/Users/alirazi/Downloads/breast cancer ultrasound.mov"
    st.video(video_path, format="video/mp4")

    

if page == "Model Testing":
    st.title("Model Testing")
    st.write("You can test the breast cancer detection model by uploading an image.")
    st.write("The model will predict the probability of malignancy.")
    
    # Load the model
    CNN_model = load_model("/Users/alirazi/BreastCancerUltrasound/custom_model.keras")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an ultrasound image", type=["jpg", "png"])

    if uploaded_image:
        left, right = st.columns(2)
        
        with left:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        with right:
            # Process and predict
            image = Image.open(uploaded_image)
            processed_image = np.array(image.resize((200, 100)).convert('L'))  # 100 x 200
            processed_image = processed_image.reshape(1, 100, 200, 1) / 255
            prediction = CNN_model.predict(processed_image)
            probability_for_malignant = prediction[0][1]




            st.subheader("Prediction Results")
            st.write(f"<span style='color:white;'>Model Malignant Probability: </span><span style='color:red;'>{round(probability_for_malignant * 100, 2)}%</span>", unsafe_allow_html=True)
            # Generate and display the pie chart
            labels = ['Benign', 'Malignant']
            sizes = [1 - probability_for_malignant, probability_for_malignant]
            colors = ['#66b3ff', '#ff9999']
            fig, ax = plt.subplots(facecolor="black")
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)
