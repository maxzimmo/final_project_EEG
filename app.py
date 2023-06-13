import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models


# load model and predict after css!!!

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the model once instead of every time the function is called
model = models.load_model("./model.h5")

def predict(input_file):
    '''needs a file path as input. Loads .npy file and predicts. Returns prediction'''
    X_test = np.load(input_file)
    prediction = model.predict(X_test)
    return prediction


def main():
    load_css('style.css')
    
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://federalnewsnetwork.com/wp-content/uploads/2023/05/GettyImages-1460853312-1880x1057.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
    
    st.write("""
        <div class="container d-flex flex-column justify-content-center align-items-center vh-100">
            <h1 class="text-center p-2 display-4 mt-5 eeg-title">E.E.G</h1>
            <p class="text-center eeg-caption">Electroencephalogram</p>
            <div class="eeg-description">
                <p>
                    Welcome! To get a prediction for your brain state, please upload an EEG (Electroencephalogram) file in npz
                    format.
                    This will allow us to analyze your brain activity and provide insights. Simply drag and drop the file into the
                    designated area or click to upload.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Create a directory for file uploads
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)

    # Upload file
    uploaded_file = st.file_uploader("Choose a .npy file", type=['npy'])

    # Check if file was uploaded
    if uploaded_file is not None:
        # Save the uploaded file to the upload directory
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getvalue())
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

        # Run prediction
        output = predict(file_path)
        final_trial_predict = np.max(output[0])
        st.markdown(f"**Prediction Result:** {final_trial_predict}")
        print(output)
        st.write(output)
        plt.plot(output)
        st.pyplot(plt)


if __name__ == '__main__':
    main()
