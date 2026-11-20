import sys
sys.setrecursionlimit(10000)

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import time

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="Human Age Group Detection",
    page_icon="👤",
    layout="centered"
)

st.title("👤 Human Age Group Detection")
st.write("Detect Baby, Kid, Teenager, or Adult from an image, a photo, or live webcam.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return YOLO("best (2).pt")

model = load_model()

# ---------------- Choose Input ----------------
option = st.radio(
    "Choose input:",
    ["Upload Image", "Take Picture", "Live Camera"]
)

# ---------------- 1. UPLOAD IMAGE ----------------
if option == "Upload Image":
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("Detecting..."):
            results = model.predict(source=image, conf=0.25, verbose=False)
            result_image = results[0].plot()

        st.subheader("Detection Result")
        st.image(result_image, channels="BGR", use_container_width=True)

        for box in results[0].boxes:
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"- **{name}** - {conf:.1%}")

# ---------------- 2. TAKE PICTURE ----------------
elif option == "Take Picture":
    camera = st.camera_input("Take a picture")
    if camera is not None:
        image = Image.open(camera)
        st.image(image, caption="Captured Image", use_container_width=True)

        with st.spinner("Detecting..."):
            results = model.predict(source=image, conf=0.25, verbose=False)
            result_image = results[0].plot()

        st.subheader("Detection Result")
        st.image(result_image, channels="BGR", use_container_width=True)

        for box in results[0].boxes:
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"- **{name}** - {conf:.1%}")

# ---------------- 3. LIVE CAMERA ----------------
elif option == "Live Camera":
    st.subheader("Live Camera")
    st.write("Click Start to open your webcam. Click Stop to close it.")
    
    start = st.button("▶️ Start Live Camera")
    stop = st.button("⏹️ Stop")

    frame_placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)  # 0 = your laptop webcam

        if not cap.isOpened():
            st.error("Could not open webcam. Check if another app is using it.")
        else:
            # This flag lets us stop
            st.session_state.run_live = True

            while st.session_state.run_live:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                # Run YOLO on the frame
                results = model.predict(source=frame, conf=0.25, verbose=False)
                annotated = results[0].plot()

                # Show it (convert BGR to RGB for Streamlit)
                frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

                # Small delay so it doesn't overload
                time.sleep(0.05)

                # If user clicks Stop, Streamlit will rerun and we stop
                if stop:
                    st.session_state.run_live = False
                    break

            cap.release()
