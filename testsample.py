import cv2 
import numpy as np
import yoloposevideo
import matplotlib.pyplot as plt

count = 0
points = []
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

if __name__=="__main__": 
	pts2 = np.float32([[0,0],[0,720],[1280,0],[1280,720]])
	vid = cv2.VideoCapture("airport.mp4")
	success, image = vid.read()
	while success:
		success, image =vid.read()
		if success == False:
			continue
		cv2.imshow('image', image)
		if count < 4:  
			cv2.setMouseCallback('image', click_event)
		else:
			transformed_frame = bird_eye_tranform(image, points, pts2) 
			out.write(transformed_frame)
			cv2.imshow('trans', transformed_frame)
		cv2.waitKey(34)
	out.release()
	cv2.destroyAllWindows() 
	if count ==4:
		yoloposevideo.main_code()
