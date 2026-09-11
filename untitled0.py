import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load your trained YOLO model
model = YOLO("best (2).pt")

# Page title
st.title("Human Age Group Detection")

st.write("Upload an image or take a picture using your camera.")

# Choose input method
input_method = st.radio(
    "Choose how you want to provide an image:",
    ["Upload Image", "Use Camera"]
)

# -------------------------------
# OPTION 1: UPLOAD IMAGE
# -------------------------------
if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Open uploaded image
        image = Image.open(uploaded_file)

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        # Run YOLO
        results = model.predict(
            source=image,
            conf=0.25
        )

        # Draw bounding boxes
        result_image = results[0].plot()

        st.subheader("Detection Result")

        st.image(
            result_image,
            channels="BGR",
            use_container_width=True
        )


# -------------------------------
# OPTION 2: CAMERA
# -------------------------------
elif input_method == "Use Camera":

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:

        # Open camera image
        image = Image.open(camera_image)

        st.subheader("Captured Image")
        st.image(image, use_container_width=True)

        # Run YOLO
        results = model.predict(
            source=image,
            conf=0.25
        )

        # Draw bounding boxes
        result_image = results[0].plot()

        st.subheader("Detection Result")

        st.image(
            result_image,
            channels="BGR",
            use_container_width=True
        )