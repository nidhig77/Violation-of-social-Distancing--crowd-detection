


import numpy as np
from ultralytics import YOLO
import cv2
import statistics

def homography(avg_height):
    avg_tot = []
    f = cv2.VideoCapture('transformed_video.mp4')
    _, image = f.read()
    cv2.imwrite("frame.jpg",image)
    model = YOLO('yolov8m-pose.pt')
    results = model("frame.jpg")
    pose=[]
    for r in results:
        no_of_persons = r.keypoints.xy.size(0)
        keypoint_data = r.keypoints
        for _ in range(int(no_of_persons)):
            posefeats =[]
            head_point = keypoint_data.xy[_][0]
            right_point = keypoint_data.xy[_][16]
            if (head_point == 0).any() or (right_point == 0).any():
                continue

            posefeats.append(head_point)
            posefeats.append(right_point)
            pose.append(posefeats)
            posefeats =[]
    cordarray = np.array(pose)
    for x in range(len(cordarray)):
        dist = np.linalg.norm(cordarray[x][0] - cordarray[x][1])
        avg_tot.append(dist)
    
    avg=statistics.mean(avg_tot)
    vioDist = (avg/avg_height)*1.25
    warnDist = (avg/avg_height)*2.5
    return vioDist,warnDist



import cv2 
import numpy as np
import pose_extractor
import matplotlib.pyplot as plt
from homography import homography

count = 0
points = []
out = cv2.VideoWriter('transformed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (1280, 720))
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
	avg_height = float(input("Enter average height of citizens:"))
	pts2 = np.float32([[0,0],[0,720],[1280,0],[1280,720]])
	vid = cv2.VideoCapture("E:\projectex\crowd_footpath.mp4")
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
		cv2.imshow('image', image)
		points= [
            (0, 0),
            (0, image.shape[0]),
            (image.shape[1], 0),
            (image.shape[1], image.shape[0]),
        ]
		print(points)
		transformed_frame = bird_eye_tranform(image, points, pts2) 
		out.write(transformed_frame)
		cv2.imshow('trans', transformed_frame)
		cv2.waitKey(34)
	out.release()
	cv2.destroyAllWindows() 
	
	vioLength, warnLength = homography(avg_height)
	vioLength = int(vioLength//1)
	warnLength = int(warnLength//1)
	pose_extractor.main_code(vioLength, warnLength)


import numpy as np
from ultralytics import YOLO
import cv2
import statistics

def homography(avg_height):
    distance_between_two=[] #contains the list of distances between all the people
    f = cv2.VideoCapture('transformed_video.mp4')
    _, image = f.read()
    cv2.imwrite("frame.jpg",image)
    model = YOLO('yolov8m-pose.pt')
    results = model("frame.jpg")
    pose=[]
    for r in results:
        no_of_persons = r.keypoints.xy.size(0)
        keypoint_data = r.keypoints
        for _ in range(int(no_of_persons)):
            posefeats =[]
            head_point = keypoint_data.xy[_][0]
            right_point = keypoint_data.xy[_][16]
            if (head_point == 0).any() or (right_point == 0).any():
                continue

            posefeats.append(head_point)
            posefeats.append(right_point)
            pose.append(posefeats)
            posefeats =[]
    cordarray = np.array(pose)
    for x in range(len(cordarray)):
        for i in range( x +1 , len(cordarray)):
            dist_head = np.linalg.norm(cordarray[x][0]-cordarray[i][0])
            dist_right = np.linalg.norm(cordarray[x][1]-cordarray[i][1])
            avg_between_two= (dist_head+dist_right)/2 #distance between two people
            distance_between_two.append(avg_between_two)
    
    avg=statistics.mean(distance_between_two)
    print("avg:",avg)
    vioDist = (avg/avg_height)*1.5
    warnDist = (avg/avg_height)*2.5
    print("viodist:",vioDist)
    return vioDist,warnDist

import numpy as np
from ultralytics import YOLO
import cv2
import statistics

def homography(avg_height):
    distance_between_two=[] #contains the list of distances between all the people
    f = cv2.VideoCapture('transformed_video.mp4')
    _, image = f.read()
    cv2.imwrite("frame.jpg",image)
    model = YOLO('yolov8m-pose.pt')
    results = model("frame.jpg")
    pose=[]
    for r in results:
        no_of_persons = r.keypoints.xy.size(0)
        keypoint_data = r.keypoints
        for _ in range(int(no_of_persons)):
            posefeats =[]
            head_point = keypoint_data.xy[_][0]
            right_point = keypoint_data.xy[_][16]
            if (head_point == 0).any() or (right_point == 0).any():
                continue

            posefeats.append(head_point)
            posefeats.append(right_point)
            pose.append(posefeats)
            posefeats =[]
    cordarray = np.array(pose)
    for x in range(len(cordarray)):
        for i in range( x +1 , len(cordarray)):
            dist_head = np.linalg.norm(cordarray[x][0]-cordarray[i][0])
            dist_right = np.linalg.norm(cordarray[x][1]-cordarray[i][1])
            avg_between_two= (dist_head+dist_right)/2 #distance between two people
            distance_between_two.append(avg_between_two)
    
    avg=statistics.mean(distance_between_two)
    print("avg:",avg)
    vioDist = (avg/avg_height)*1.5
    warnDist = (avg/avg_height)*2.5
    print("viodist:",vioDist)
    return vioDist,warnDist



