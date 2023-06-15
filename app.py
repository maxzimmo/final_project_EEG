import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Load the model once instead of every time the function is called
model = models.load_model("./data/models/model.h5")
# Define your label dictionary
label_dict = {0: 'Disgust', 1: 'Fear', 2: 'Sad', 3: 'Neutral', 4: 'Happy'}

image_dict = {0: './data/emotion_images/disgust.png',
              1: './data/emotion_images/fear.png',
              2: './data/emotion_images/sad.png',
              3: './data/emotion_images/neutral.png',
              4: './data/emotion_images/happy.png'}


def predict(input_file):
    '''needs a file path as input. Loads .npy file and predicts. Returns prediction'''
    with st.spinner('Predicting...'):
        X_test = np.load(input_file)
        prediction = model.predict(X_test)
    return prediction


def main():

    # Load y_test data
    y_test = np.load('./data/models/y_test.npy')

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
            <p class="text-center eeg-caption">The Emotion Predictor</p>
            <div class="eeg-description">
                <p>
                    Welcome to the emotion predictor! Upload a file to get a prediction
                    This will allow us to analyze your brain activity and provide insights. Simply drag and drop the file into the
                    designated area or click to upload.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create a directory for file uploads
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)
    
    #---------------------
    
    # Upload file
    uploaded_file = st.file_uploader("Choose a .npy file", type=['npy'])

    default_file_button = False
    if uploaded_file is None:
        default_file_button = st.button("*")

    # Check if file was uploaded or default file button was clicked
    if uploaded_file is not None or default_file_button:
        if uploaded_file is not None:
            # Save the uploaded file to the upload directory
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())
            st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        elif default_file_button:
            # Use default file
            file_path = './data/models/X_test.npy'

        # Get file summary
        X_test = np.load(file_path)
    #---------------------

        #st.info(
        #    f"The uploaded file contains {X_test.shape[0]} sets to be predicted.")
        st.markdown(f"""
            <div style='background-color:rgba(0, 0, 255, 0.5);'>
                <p>The uploaded file contains {X_test.shape[0]} moviesets to be predicted.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Run prediction
        output = predict(file_path)
        st.success("Congratulations, now we have predictions!")

        # Use a dropdown to select the observation to display
        selected_observation = st.selectbox(
            'Choose a set to display:', options=range(len(output)), 
            index=7
            )

        # Get prediction for selected observation
        selected_output = output[selected_observation]
        
        # Get the prediction category with the highest probability
        best_category = np.argmax(selected_output)


        col1, col2 = st.columns([1, 1])

        # Display the markdown text in the second column
        col1.markdown(
            f"## **The predicted emotion is:**\n <span class='emotion-label-result'>{label_dict[best_category]}</span>",
            unsafe_allow_html=True
        )
        
        # Display the image in the first column
        col2.image(image_dict[best_category], use_column_width=True)

        predicted_acc = max(selected_output)*100
        formatted_string = "{:.2f}".format(predicted_acc)
        # format to two decimal places
        float_value = float(formatted_string)

        # Summary Analysis
        st.markdown(
            f'''For this set, the most likely emotion is '{label_dict[best_category]}', with a probability of: {float_value} %''')

        # Create a bar plot for the probabilities of each category
        fig, ax = plt.subplots()
        sns.set_theme(style='darkgrid', rc={'figure.figsize':(12,6)})
        sns.barplot(x=list(label_dict.values()), y=selected_output, ax=ax)
        ax.set_title('Emotion Probability Distribution', fontsize=40)
        ax.set_xlabel('Emotions', fontsize=20)
        ax.set_ylabel('Probability', fontsize=20)
        st.pyplot(fig)



if __name__ == '__main__':
    main()
