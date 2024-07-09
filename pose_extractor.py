import pandas as pd
from ultralytics import YOLO
import numpy as np
import heatmap
import os

model = YOLO('yolov8m-pose.pt')
violationsVsFrame = []
warningsVsFrame = []

def extract_poses(frame):
    # Run pose estimation on the frame
    results = model(frame)
    poses = []

    for r in results:
        no_of_persons = r.keypoints.xy.size(0)
        keypoint_data = r.keypoints
        for _ in range(int(no_of_persons)):
            head_point = keypoint_data.xy[_][0]
            right_point = keypoint_data.xy[_][16]
            if (head_point == 0).any() or (right_point == 0).any():
                continue
            pose_center = (int((head_point[0] + right_point[0]) / 2), int((head_point[1] + right_point[1]) / 2))
            poses.append(pose_center)

    return poses

def main_code(violationData, warningData):
    if os.path.exists('file.csv'):
        os.remove('file.csv')
    if os.path.exists('file2.csv'):
        os.remove('file2.csv')

    results = model("transformed_video.mp4", show=True, stream=True)
    for r in results:
        no_of_persons = r.keypoints.xy.size(0)
        keypoint_data = r.keypoints
        pose = []
        posefeathead = []

        for _ in range(int(no_of_persons)):
            posefeats = []
            head_point = keypoint_data.xy[_][0]
            right_point = keypoint_data.xy[_][16]
            if (head_point == 0).any() or (right_point == 0).any():
                continue

            posefeats.append(head_point)
            posefeathead.append(head_point)
            posefeats.append(right_point)
            pose.append(posefeats)
            posefeats = []

        cordarray = np.array(posefeathead)
        print(cordarray)

        cor = np.zeros([720, 1280], dtype=int)
        for i in range(len(cordarray)):
            cor[int(cordarray[i][1] // 1)][int(cordarray[i][0] // 1)] = 1

        violations, warnings = heatmap.heatmap(cor, warningData, violationData)
        violationsVsFrame.append(violations)
        warningsVsFrame.append(warnings)
        vioDict = {'vio': violationsVsFrame}
        warnDict = {'warn': warningsVsFrame}
        df = pd.DataFrame(vioDict)
        df2 = pd.DataFrame(warnDict)
        df.to_csv('file.csv')
        df2.to_csv('file2.csv')
