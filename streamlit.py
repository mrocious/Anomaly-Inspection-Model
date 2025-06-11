import streamlit as st
from PIL import Image
import os
from predict_tf import predict_image

# Set up the page layout
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")
st.title("InspectorsAlly")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)

st.write(
    "Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly."
)

with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections. With InspectorsAlly, companies can ensure that their products meet the highest standards of quality, while reducing inspection time and increasing efficiency."
    )
    st.write(
        "This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. InspectorsAlly is capable of identifying even the slightest defects, such as scratches, dents, discolorations, and more on the Leather Product Images."
    )

# Define the function to load images
def load_uploaded_image(file):
    img = Image.open(file)
    return img

# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio(
    "options", ["File Uploader", "Camera Input"], label_visibility="collapsed"
)

# Load image
uploaded_file_img = None
camera_file_img = None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# Predict
submit = st.button(label="Submit a Product Image")
if submit:
    st.subheader("Output")
    img_file = uploaded_file_img if input_method == "File Uploader" else camera_file_img
    if img_file is None:
        st.error("No image provided.")
    else:
        with st.spinner(text="Running prediction..."):
            prediction = predict_image(img_file)
            st.write(prediction)
