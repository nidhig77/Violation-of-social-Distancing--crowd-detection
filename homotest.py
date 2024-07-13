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
	vid = cv2.VideoCapture("crowd_mall.mp4")
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


# Example usage:
# Replace this with your actual coordinates array
#sample_coordinates = [[(100,150),( 200,300)], [(300,345),( 400,340)]]
#show_violators(sample_coordinates)
#stframe.image(frame, channels="BGR")