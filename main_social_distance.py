import cv2 
import numpy as np
import pose_extractor
import matplotlib.pyplot as plt
from homography import homography
from show_violators import show_violators
import streamlit as st
import tempfile
import time
st.title("Social Distancing violator detector")
avg_height = st.number_input(
    "Enter average height of citizens (in meters):",
    min_value=0.1,
    max_value=3.0,
    step=0.1,
)

uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov", "mkv"]
)
count = 0
points = []
frame_count = 0
stframe=st.empty()
out = cv2.VideoWriter('transformed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
def click_event(event, x, y, flags, params): 
	global count
	if event == cv2.EVENT_LBUTTONDOWN: 
		print(x, ' ', y) 
		points.append((x,y))
		count+=1

def bird_eye_tranform(image, points, pts2):
	points = np.float32(points)
	matrix = cv2.getPerspectiveTransform(points, pts2)
	transformed_frame = cv2.warpPerspective(image, matrix, (1280, 720))
	return transformed_frame

if uploaded_file is not None and avg_height > 0:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
else:
      st.write("Upload a video")
if uploaded_file is not None and avg_height > 0:
	pts2 = np.float32([[0,0],[0,720],[1280,0],[1280,720]])
	vid = cv2.VideoCapture(tfile.name)
	total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	print(total_frames)
	success, image = vid.read()
	count = 5
	while success:
		success, image =vid.read()
		count=count+5
		vid.set(cv2.CAP_PROP_POS_FRAMES, count)
		if success == False:
			continue
		#cv2.imshow('image', image)
		points= [
            (0, 0),
            (0, image.shape[0]),
            (image.shape[1], 0),
            (image.shape[1], image.shape[0]),
        ]
	#print(points)
		transformed_frame = bird_eye_tranform(image, points, pts2) 
		out.write(transformed_frame)
		#cv2.imshow('trans', transformed_frame)
		cv2.waitKey(34)

	#cv2.destroyAllWindows() 
	out.release()

	vioLength, warnLength = homography(avg_height)
	vioLength = int(vioLength//1)
	warnLength = int(warnLength//1)
	violators_for_each_frame= pose_extractor.main_code(vioLength, warnLength)
	#show_violators(violators_for_each_frame )
	cap = cv2.VideoCapture("transformed_video.mp4")
	if not cap.isOpened():
		st.write("Error opening this file")
	else:
		ret,frame = cap.read()
		while ret:
			ret,frame=cap.read()
			frame_count+=1
			if not ret:
				break
			if violators_for_each_frame[frame_count]:
				for x,y in violators_for_each_frame[frame_count]:
					# Draw a red bounding box around the specified coordinates
					cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2)
			stframe.image(frame, channels="BGR")
			time.sleep(0.3)
			if cv2.waitKey(30) & 0xFF == ord("q"):
				break

		cap.release()
