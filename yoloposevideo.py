import re
from ultralytics import YOLO
import numpy as np
import pprint
import heatmap
import torch
import time

import distance_estimation as de
from test import transform


model = YOLO('yolov8m-pose.pt')
violationsVsFrame = []
# import cv2

# # Open the input video file
# input_file = 'resizedcrowd_mall1.mp4'
# cap = cv2.VideoCapture(input_file)

# # Define the output video file
# output_file = 'resizedcrowd_mall1.mp4'

# # Get the frame dimensions of the input video
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the target dimensions (640x640)
# target_width, target_height = 1280, 720

# # Create VideoWriter object to save the resized video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_file, fourcc, 15.0, (target_width, target_height))

# # Resize each frame and write it to the output video
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Resize the frame to the target dimensions
#     resized_frame = cv2.resize(frame, (target_width, target_height))

#     # Write the resized frame to the output video
#     out.write(resized_frame)

# # Release video capture and writer objects
# cap.release()
# out.release()

# print("Resized video saved successfully!")
# test = False
# test=transform.transm("crowd_mall1.mp4")
# print("done")


def main_code():
    results = model("transformed_video.mp4", show=True,stream=True)
    for r in results:
        no_of_persons = r.keypoints.xy.size(0)
    # print(f"No of persons detected {int(no_of_persons)}")

        keypoint_data = r.keypoints
        box_data = r.boxes


        
        
        pose=[]

        posefeathead = []


        for _ in range(int(no_of_persons)):
            posefeats =[]
            head_point = keypoint_data.xy[_][0]
            right_point = keypoint_data.xy[_][16]
            if (head_point == 0).any() or (right_point == 0).any():
                continue  # Skip this tensor if it contains any 0 values

            posefeats.append(head_point)
            posefeathead.append(head_point)
            posefeats.append(right_point)
            pose.append(posefeats)
            # x = torch.any(head_point==0)
            # if x is True:
            #     continue
            # else:
            # posefeats.append(head_point)
            # posefeathead.append(head_point)
            # # print(head_point)
            
            # y = torch.any(right_point==0)
            # # if y is True:
            # #     continue
            # # else:
            # posefeats.append(right_point)
            # print(posefeats)
            # pose.append(posefeats)
            posefeats =[]

        posenp = np.array(pose) 
        # filtered_tensors = [tensor for tensor in posefeats if not torch.any(tensor == 0)]
        # print(filtered_tensors)
        
        cordarray = np.array(posefeathead)
        print(cordarray)
        # print(np.shape(cordarray))
        # print(len(cordarray))
        # pprint.pprint(posenp) 
            
        
    #     no = de.get_respect_social_distancing(posenp)
        # X,Y,_,_,_ = de.from_2D_to_3D(posenp,1280,720)
        # # print(posenp)
        # xn = np.array(X)
        # yn = np.array(Y)

        # print(xn,yn)

    # print(xn.shape)
        # xnorm = []
        # ynorm = []
        # for i in xn:
        #     if -15.0 <= i <=15.0:
        #         xnorm.append(round(i+15.0))
        
        # for i in yn:
        #     if -15.0 <= i <= 15.0:
        #         ynorm.append(round(i+15.0))

        # xnnorm =  np.array(xnorm)
        # ynnorm  = np.array(ynorm)

        # print(xn,yn)
        # print(xnorm,ynorm)
        # pprint.pprint(xnnorm)
        cor = np.zeros([720,1280], dtype=int)
        for i in range(len(cordarray)):
            cor[int(cordarray[i][1]//1)][int(cordarray[i][0]//1)]=1
            # cor[xnorm[i]][ynorm[i]]=1
        
        violationsVsFrame.append(heatmap.heatmap(cor))
            # tl_x = box_data.xyxy[_].numpy()
            # confidence = box_data.conf.numpy()

            # print(f"For Person {_+1}")
            # print(f"Head Point: {head_point}, Right Toe Point: {right_point}")
            # print("Bounding box cordinates in terms of x,y,width,height")
            # print(tl_x)
            # print(f"Bounding box confidence: {round(confidence[_].item(),2)}")

        # print(r.keypoints)
        # print(r.boxes)
    #     print(r.probs)

    # for _ in range(no_of_persons):
    #     print(f"Head and toe cordinates for {_} person")
    #     print(result_head[_][0].numpy())
    #     print(result_head[_][16].numpy())

    # head_array = result_head.numpy()
