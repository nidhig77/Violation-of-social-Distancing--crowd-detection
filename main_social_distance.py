import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

st.title("Social Distancing Detector")

# Step 1: Upload Video and Input Average Height
avg_height = st.number_input("Enter average height of citizens (in meters):", min_value=0.1, max_value=3.0, step=0.1)
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None and avg_height > 0:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    stframe = st.empty()
    points = []
    count = 0
    pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])

    def click_event(event, x, y, flags, param):
        global count
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            count += 1

    # Step 2: Select 4 Points for Perspective Transformation
    if count < 4:
        st.info("Click on the video frame to select 4 points for perspective transformation.")
        while cap.isOpened() and count < 4:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Select Points', frame)
            cv2.setMouseCallback('Select Points', click_event)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            stframe.image(frame, channels="BGR")
        
        cv2.destroyAllWindows()

    if count == 4:
        st.success("Points selected successfully!")

        def bird_eye_transform(image, points, pts2):
            points = np.float32(points)
            matrix = cv2.getPerspectiveTransform(points, pts2)
            transformed_frame = cv2.warpPerspective(image, matrix, (1280, 720))
            return transformed_frame

        def check_social_distancing(poses, threshold):
            violating_pairs = []
            for i in range(len(poses)):
                for j in range(i + 1, len(poses)):
                    dist = np.linalg.norm(np.array(poses[i]) - np.array(poses[j]))
                    if dist < threshold:
                        violating_pairs.append((poses[i], poses[j]))
            return violating_pairs

        def draw_boxes(frame, poses, violating_pairs):
            for pose in poses:
                cv2.circle(frame, pose, 5, (0, 255, 0), -1)
            for pair in violating_pairs:
                cv2.line(frame, pair[0], pair[1], (0, 0, 255), 2)
                cv2.rectangle(frame, (pair[0][0] - 10, pair[0][1] - 10), (pair[0][0] + 10, pair[0][1] + 10), (0, 0, 255), 2)
                cv2.rectangle(frame, (pair[1][0] - 10, pair[1][1] - 10), (pair[1][0] + 10, pair[1][1] + 10), (0, 0, 255), 2)

        model = YOLO('yolov8m-pose.pt')

        out = cv2.VideoWriter('transformed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            transformed_frame = bird_eye_transform(frame, points, pts2)
            results = model(transformed_frame)
            
            poses = []
            for r in results:
                for person in r.keypoints.xy:
                    head_point = person[0].cpu().numpy()
                    poses.append((int(head_point[0]), int(head_point[1])))

            violating_pairs = check_social_distancing(poses, avg_height * 100)
            draw_boxes(transformed_frame, poses, violating_pairs)
            
            out.write(transformed_frame)
            stframe.image(transformed_frame, channels="BGR")
        
        out.release()
        cap.release()

        st.success("Video processing completed and saved as 'transformed_video.mp4'.")
        st.video('transformed_video.mp4')
