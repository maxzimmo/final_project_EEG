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
label_dict = {0:'Disgust', 1:'Fear', 2:'Sad', 3:'Neutral', 4:'Happy'}

# Define your images dictionary
#image_dict = {0: 'https://media.istockphoto.com/id/627798010/photo/man-disgusted.jpg?s=612x612&w=0&k=20&c=OL6W0mUlXyn4RJKFdpBFNdqWapb5zYJO2_jbIMuErmE=',
#              1: 'https://musimorphic.com/wp-content/uploads/2022/10/home-alone-1024x768.jpg',
#              2: 'https://media.istockphoto.com/id/151557041/photo/baby-crying.jpg?s=612x612&w=0&k=20&c=PR6N_B-8TRjeyBVPzud5Gw_sJZZlf3wOgtg_4-AmGbM=',
#              3: 'https://media-cldnry.s-nbcnews.com/image/upload/t_fit-760w,f_auto,q_auto:best/newscms/2016_06/973451/guy-raised-eyebrow-today-160212.jpg',
#              4: 'https://1000wordphilosophy.files.wordpress.com/2021/05/happiness.jpg?w=640'}
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
    
    st.title("EEG Predictor")
   

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
            <p class="text-center eeg-caption">Electroencephalogram</p>
            <div class="eeg-description">
                <p>
                    Welcome to the Emotions predictor! Upload a file to get a prediction
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

        # Get file summary
        X_test = np.load(file_path)
        st.info(f"The uploaded file contains {X_test.shape[0]} sets to be predicted.")

        # Run prediction
        output = predict(file_path)
        st.success("Congratulations, now we have predictions!")

        # Use a dropdown to select the observation to display
        selected_observation = st.selectbox('## Choose an observation to display:', options=range(len(output)))
        
        # Get prediction for selected observation
        selected_output = output[selected_observation]
        
        # Get the prediction category with the highest probability
        best_category = np.argmax(selected_output)
        
        st.image(image_dict[best_category], use_column_width=True)
        st.markdown(f"## **The emotion predicted is:** {label_dict[best_category]}")

        predicted_acc = max(selected_output)*100 
        formatted_string = "{:.2f}".format(predicted_acc)
        # format to two decimal places
        float_value = float(formatted_string)
        
        #Summary Analysis
        st.markdown(f'''For this set, the most likely Emotion is '{label_dict[best_category]}', with a probability of: {float_value} %''')

        # Create a bar plot for the probabilities of each category
        fig, ax = plt.subplots()
        ax.bar(label_dict.values(), selected_output)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Probability')
        st.pyplot(fig)

        # Compute confusion matrix
        st.header('''Confusion Matrix''')
        st.write('''## How well is the model classifying emotions?''')
        st.write("Each cell in the matrix corresponds to a specific combination of the predicted and actual values. ")
        st.write("The diagonal of the matrix shows the number of correct predictions.")

        y_pred = np.argmax(output, axis=1)
        y_test_label = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_test_label, y_pred)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=label_dict.values(), yticklabels=label_dict.values(), title="Confusion matrix")
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
if __name__ == '__main__':
    main()
