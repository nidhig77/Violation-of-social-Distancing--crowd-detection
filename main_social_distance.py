import cv2
import numpy as np
import pose_extractor
import matplotlib.pyplot as plt
from homography import homography
from show_violators import show_violators
import streamlit as st
import tempfile
import time

st.title("Social Distancing Violator Detector")

avg_height = st.number_input(
    "Enter average height of citizens (in meters):",
    min_value=0.1,
    max_value=3.0,
    step=0.1,
)

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

def bird_eye_transform(image, points, pts2):
    points = np.float32(points)
    matrix = cv2.getPerspectiveTransform(points, pts2)
    transformed_frame = cv2.warpPerspective(image, matrix, (1280, 720))
    return transformed_frame

if uploaded_file is not None and avg_height > 0:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    vid = cv2.VideoCapture(tfile.name)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total frames in the video: {total_frames}")

    pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])
    out = cv2.VideoWriter('transformed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

    count = 0
    while vid.isOpened():
        vid.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, image = vid.read()
        if not success:
            break

        points = [
            (0, 0),
            (0, image.shape[0]),
            (image.shape[1], 0),
            (image.shape[1], image.shape[0]),
        ]

        transformed_frame = bird_eye_transform(image, points, pts2)
        out.write(transformed_frame)

        count += 5

    vid.release()
    out.release()

    vioLength, warnLength = homography(avg_height)
    vioLength = int(vioLength // 1)
    warnLength = int(warnLength // 1)
    violators_for_each_frame = pose_extractor.main_code(vioLength, warnLength)

    cap = cv2.VideoCapture("transformed_video.mp4")
    if not cap.isOpened():
        st.write("Error opening the transformed video file")
    else:
        stframe = st.empty()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count < len(violators_for_each_frame) and violators_for_each_frame[frame_count]:
                for x, y in violators_for_each_frame[frame_count]:
                    cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), (0, 0, 0), 7)  

            stframe.image(frame, channels="BGR")
            time.sleep(0.5)

        cap.release()
else:
    st.write("Please upload a video file and enter the average height.")
