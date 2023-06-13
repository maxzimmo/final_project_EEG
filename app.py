from keras import models

# Load the model once instead of every time the function is called
model = models.load_model("./model.h5")


def predict(input_file):
    '''needs a file path as input. Loads .npy file and predicts. Returns prediction'''
    X_test = np.load(input_file)
    prediction = model.predict(X_test)
    return prediction


def main():
    st.title("File Upload Example")

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
