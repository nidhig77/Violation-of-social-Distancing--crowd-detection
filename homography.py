import numpy as np
from ultralytics import YOLO
import cv2
import statistics


def homography(avg_height, video_path="transformed_video.mp4"):
    avg_tot = []
    f = cv2.VideoCapture(video_path)
    model = YOLO("yolov8m-pose.pt")
    pose = []

    while f.isOpened():
        ret, image = f.read()
        if not ret:
            break

        cv2.imwrite("frame.jpg", image)
        results = model("frame.jpg")

        for r in results:
            no_of_persons = r.keypoints.xy.size(0)
            keypoint_data = r.keypoints
            for _ in range(int(no_of_persons)):
                posefeats = []
                head_point = keypoint_data.xy[_][0]
                right_point = keypoint_data.xy[_][16]

                if (head_point == 0).any() or (right_point == 0).any():
                    continue

                posefeats.append(head_point)
                posefeats.append(right_point)
                pose.append(posefeats)

        cordarray = np.array(pose)

        for x in range(len(cordarray)):
            dist = np.linalg.norm(cordarray[x][0] - cordarray[x][1])
            avg_tot.append(dist)

    avg = statistics.mean(avg_tot)
    vioDist = (avg / avg_height) * 1.25
    warnDist = (avg / avg_height) * 2.5

    return vioDist, warnDist
