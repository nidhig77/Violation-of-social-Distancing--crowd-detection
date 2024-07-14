import pandas as pd
from ultralytics import YOLO
import numpy as np
import heatmap
import os
import streamlit as st
#from show_violators import show_violators
model = YOLO('yolov8m-pose.pt')
violationsVsFrame = []
warningsVsFrame = []
placeholder=st.empty()

def main_code(violationData, warningData):
    if(os.path.exists('file.csv')):
        os.remove('file.csv')
    if(os.path.exists('file2.csv')):
        os.remove('file2.csv')
    results = model("transformed_video.mp4", show=False,stream=True)
    violators_for_each_frame = []
    for r in results:
        no_of_persons = r.keypoints.xy.size(0)

        keypoint_data = r.keypoints
     
        pose=[]

        posefeathead = []


        for _ in range(int(no_of_persons)):
            posefeats =[]
            head_point = keypoint_data.xy[_][0]
            right_point = keypoint_data.xy[_][16]
            if (head_point == 0).any() or (right_point == 0).any():
                continue

            posefeats.append(head_point)
            posefeathead.append(head_point)
            posefeats.append(right_point)
            pose.append(posefeats)
            posefeats =[]
        
        cordarray = np.array(posefeathead)
        print(cordarray)


        cor = np.zeros([720,1280], dtype=int)
        for i in range(len(cordarray)):
            cor[int(cordarray[i][1]//1)][int(cordarray[i][0]//1)]=1
        
        violators, violations, warnings=(heatmap.heatmap(cor,warningData,violationData))
        violators_for_each_frame.append(violators)
        violationsVsFrame.append(violations)
        warningsVsFrame.append(warnings)
        vioDict = {'vio': violationsVsFrame}
        warnDict = {'warn':warningsVsFrame}
        df = pd.DataFrame(vioDict)
        df2 =pd.DataFrame(warnDict)
        df.to_csv('file.csv')
        df2.to_csv('file2.csv')
    return violators_for_each_frame
    


           

   
