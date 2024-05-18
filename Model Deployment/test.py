import streamlit as st
import tensorflow
from ultralytics import YOLO
from tensorflow.keras.models import load_model # type: ignore
from utilitypy import *
from ultralytics.utils.plotting import Annotator
from itertools import chain
import os

ppe_detection = YOLO(r'Models\best_construction.pt')
haar_model = load_model(r'Models\haar_model.h5')
skeleton_detection = YOLO('yolov8n-pose.pt')


def process_video():
    # This function will be called when the "Process Video" button is clicked
    # You can replace this with your actual processing logic
    st.write("Processing video...")
    # Example: Get the directory path of the uploaded video
    video_path = r"C:\Users\MJ\projects\DATASCI3\uploaded_video"
    st.write("Video path:", video_path)


# CSS for styling
st.markdown(
    """
    <style>
        .title {
            font-family: 'Arial', monospace;
            font-size: 40px;
            text-align: center;
        }
        .hehe {
            font-family: 'Arial', monospace;
            font-size: 24px;
            text-align: center;
        }
        .left {
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Structure
st.markdown("<h1 class='title'>PPE Detection and Construction Worker Movement Recognition</h1>", unsafe_allow_html=True)
st.markdown('----')

cols = st.columns(2)

uploaded = False  # Flag to track upload status
save_dir = r'C:\Users\MJ\projects\DATASCI3\uploaded_video'

with cols[0]:
    pose_uploaded_file = st.file_uploader("PPE Detection", type=["mp4"], key= 'haar_uplod')
    pose_submit_button = st.button("Run", key = 'haar_button')

    if pose_submit_button:
        if pose_uploaded_file is not None: 
            st.markdown('File Uploaded')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "pose_uploaded_video.mp4")
            with open(save_path, 'wb') as f:
                f.write(pose_uploaded_file.getbuffer())

            detect_ppe(r'uploaded_video\pose_uploaded_video.mp4', ppe_detection)
            st.video(f'output_video/pose_output_video.mp4')
        else:
            st.warning("Please upload a video file.")


with cols[1]:
    haar_uploaded_file = st.file_uploader("Activity Recognition", type=["mp4"], key = "pose_upload")
    haar_submit_button = st.button("Run", key = "pose_button")

    if haar_submit_button:
        if haar_uploaded_file is not None: 
            st.markdown('File Uploaded')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "haar_uploaded_video.mp4")
            with open(save_path, 'wb') as f:
                f.write(haar_uploaded_file.getbuffer())

            haar_recognition(r'uploaded_video\haar_uploaded_video.mp4', haar_model, skeleton_detection)
            st.video('output_video/haar_output_video.mp4')
        else:
            st.warning("Please upload a video file.")
        

# Display uploaded status
if uploaded:
    st.markdown('Uploaded')
