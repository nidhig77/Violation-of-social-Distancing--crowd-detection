import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# Define a function to show the login page
def show_login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password":  # Replace with your credentials
            st.session_state["logged_in"] = True
            st.experimental_rerun()  # Rerun the app to show the detector
        else:
            st.error("Invalid username or password")

# Define a function to show the social distancing detector
def show_detector():
    st.title("Social Distancing Detector")

    # Add logout button
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()  # Rerun the app to show the login page

    # Step 1: Upload Video and Input Average Height
    avg_height = st.number_input(
        "Enter average height of citizens (in meters):",
        min_value=0.1,
        max_value=3.0,
        step=0.1,
    )
    uploaded_file = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None and avg_height > 0:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read the video file.")
            cap.release()
        else:
            # Automatically select the end points of the video as the 4 coordinates
            points = [
                (0, 0),
                (0, frame.shape[0]),
                (frame.shape[1], 0),
                (frame.shape[1], frame.shape[0]),
            ]
            pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])

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

            def draw_boxes(frame, poses, violating_pairs, clusters):
                for i, pose in enumerate(poses):
                    color = (0, 255, 0) if i not in clusters else (255, 0, 0)
                    cv2.circle(frame, pose, 5, color, -1)
                    cv2.rectangle(
                        frame,
                        (pose[0] - 10, pose[1] - 10),
                        (pose[0] + 10, pose[1] + 10),
                        color,
                        2,
                    )
                for pair in violating_pairs:
                    cv2.line(frame, pair[0], pair[1], (0, 0, 255), 2)

            def find_clusters(poses, threshold, min_cluster_size):
                clusters = []
                for i, pose in enumerate(poses):
                    cluster = [i]
                    for j, other_pose in enumerate(poses):
                        if i != j:
                            dist = np.linalg.norm(np.array(pose) - np.array(other_pose))
                            if dist < threshold:
                                cluster.append(j)
                    if len(cluster) >= min_cluster_size:
                        clusters.extend(cluster)
                return set(clusters)

            model = YOLO("yolov8m-pose.pt")

            out = cv2.VideoWriter(
                "transformed_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720)
            )
            violations_count = 0
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                transformed_frame = bird_eye_transform(frame, points, pts2)
                results = model(transformed_frame)

                poses = []
                for r in results:
                    for person in r.keypoints.xy:
                        if person.shape[0] > 0:  # Check if person array is not empty
                            head_point = person[0].cpu().numpy()
                            poses.append((int(head_point[0]), int(head_point[1])))

                violating_pairs = check_social_distancing(poses, avg_height * 100)
                violations_count += len(violating_pairs)

                clusters = find_clusters(poses, avg_height * 100, min_cluster_size=2)
                draw_boxes(transformed_frame, poses, violating_pairs, clusters)

                out.write(transformed_frame)
                stframe.image(transformed_frame, channels="BGR")

            out.release()
            cap.release()

            avg_violations_per_frame = violations_count / frame_count
            st.success(
                f"Video processing completed with an average of {avg_violations_per_frame:.2f} violations per frame."
            )

# Check if the user is logged in
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Show the appropriate page based on the login state
if st.session_state["logged_in"]:
    show_detector()
else:
    show_login()
