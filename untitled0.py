import streamlit as st
from ultralytics import YOLO
from PIL import Image
import av
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# Load your trained YOLO model
model = YOLO("best (2).pt")


# Page settings
st.set_page_config(
    page_title="Human Age Group Detection",
    page_icon="👤",
    layout="wide"
)

st.title("Human Age Group Detection")

st.write(
    "Detect human age groups using an image, a captured camera picture, "
    "or a live camera feed."
)


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = model
        self.lock = threading.Lock()

    def recv(self, frame):

        # Convert camera frame to image
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        with self.lock:
            results = self.model.predict(
                source=img,
                conf=0.25,
                verbose=False
            )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# ============================================================
# CHOOSE INPUT METHOD
# ============================================================

input_method = st.radio(
    "Choose how you want to provide your input:",
    [
        "Upload Image",
        "Take Picture",
        "Live Camera"
    ]
)


# ============================================================
# OPTION 1: UPLOAD IMAGE
# ============================================================

if input_method == "Upload Image":

    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

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


# ============================================================
# OPTION 2: TAKE A PICTURE
# ============================================================

elif input_method == "Take Picture":

    st.subheader("Take a Picture")

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        image = Image.open(camera_image)

        st.subheader("Captured Image")

        st.image(
            image,
            use_container_width=True
        )

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


# ============================================================
# OPTION 3: LIVE CAMERA
# ============================================================

elif input_method == "Live Camera":

    st.subheader("Live Camera Detection")

    st.write(
        "Allow camera access when your browser asks for permission. "
        "YOLO will process the camera feed continuously."
    )

    webrtc_streamer(
        key="human-age-detection",

        video_processor_factory=YOLOVideoProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )

    st.info(
        "The live camera uses the same trained best (2).pt model "
        "as the image detector."
    )