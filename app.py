import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models

# Load the model once instead of every time the function is called
model = models.load_model("./model.h5")

# Define your label dictionary
label_dict = {0:'Disgust', 1:'Fear', 2:'Sad', 3:'Neutral', 4:'Happy'}

# Define your images dictionary
image_dict = {0: 'https://media.istockphoto.com/id/627798010/photo/man-disgusted.jpg?s=612x612&w=0&k=20&c=OL6W0mUlXyn4RJKFdpBFNdqWapb5zYJO2_jbIMuErmE=',
              1: 'https://musimorphic.com/wp-content/uploads/2022/10/home-alone-1024x768.jpg',
              2: 'https://media.istockphoto.com/id/151557041/photo/baby-crying.jpg?s=612x612&w=0&k=20&c=PR6N_B-8TRjeyBVPzud5Gw_sJZZlf3wOgtg_4-AmGbM=',
              3: 'https://media-cldnry.s-nbcnews.com/image/upload/t_fit-760w,f_auto,q_auto:best/newscms/2016_06/973451/guy-raised-eyebrow-today-160212.jpg',
              4: 'https://1000wordphilosophy.files.wordpress.com/2021/05/happiness.jpg?w=640'}

def predict(input_file):
    '''needs a file path as input. Loads .npy file and predicts. Returns prediction'''
    with st.spinner('Predicting...'):
        X_test = np.load(input_file)
        prediction = model.predict(X_test)
    return prediction

def main():
    st.title("EEG Predictor")
    st.markdown('''
    ## Welcome to the EEG predictor! 
    Upload a file to get a prediction!''')

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
        
        st.markdown(f"**The emotion predicted is:** {label_dict[best_category]}")
        st.image(image_dict[best_category], use_column_width=True)
        
        predicted_acc = max(selected_output)*100 
        formatted_string = "{:.2f}".format(predicted_acc)
        # format to two decimal places
        float_value = float(formatted_string)
        
        #Summary Analysis
        st.markdown(f'''For this set, the most likely Emotion is "{label_dict[best_category]}", given the highest probability:   {float_value} %''')
       


        # Create a bar plot for the probabilities of each category
        fig, ax = plt.subplots()
        ax.bar(label_dict.values(), selected_output)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Probability')
        st.pyplot(fig)
        
      
        
if __name__ == '__main__':
    main()
