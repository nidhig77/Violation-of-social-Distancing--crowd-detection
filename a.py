import cv2
cap = cv2.VideoCapture("transformed_video.mp4")
if not cap.isOpened():
	print("Error opening this file")