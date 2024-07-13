import cv2
import numpy as np
import time
import streamlit as st
def show_violators(violators_for_each_frame):
    """
    Displays a video with bounding boxes around specified coordinates.

    Args:
        coordinates_array (list of lists): A 2D array where each row represents coordinates (x, y).
        video_path (str): Path to the video file (default: "transformed_video.mp4").
    """
    cap = cv2.VideoCapture("transformed_video.mp4")
    frame_count = 0
    stframe=st.empty()
    if not cap.isOpened():
        print("Error opening video file.")
        return
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        frame_count+=1
        if not ret:
            break  # End of video
        if violators_for_each_frame[frame_count]:
            for x,y in violators_for_each_frame[frame_count]:
            # Draw a red bounding box around the specified coordinates
                cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2)
        stframe.image(frame, channels="BGR")
        time.sleep(0.5)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# Replace this with your actual coordinates array
#sample_coordinates = [[(100,150),( 200,300)], [(300,345),( 400,340)]]
#show_violators(sample_coordinates)
