import streamlit as st
import os


def add_bg_from_url():
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
# add_bg_from_url()


def main():
    st.title("File Upload Example")

    # Create a directory for file uploads
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)

    # Upload file
    uploaded_file = st.file_uploader("Choose a file")

    # Check if file was uploaded
    if uploaded_file is not None:
        # Save the uploaded file to the upload directory
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getvalue())

        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        # st.info("File saved to 'upload' directory.")
        st.markdown(f"**Uploaded File Name:** {uploaded_file.name}")
        # st.markdown(f"**File Size:** {uploaded_file.size} bytes")


if __name__ == '__main__':
    main()
