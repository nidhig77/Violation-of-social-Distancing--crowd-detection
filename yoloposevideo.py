import pandas as pd
from ultralytics import YOLO
import numpy as np
import heatmap
import os
import socialdist

model = YOLO('yolov8m-pose.pt')
violationsVsFrame = []
warningsVsFrame = []

# print("Resized video saved successfully!")
# test = False
# test=transform.transm("crowd_mall1.mp4")
# print("done")


def main_code():
    warningData = int(input("Enter warning threshold:"))
    violationData = int(input("Enter violation threshold:"))
    if(os.path.exists('file.csv')):
        os.remove('file.csv')
    if(os.path.exists('file2.csv')):
        os.remove('file2.csv')
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
                continue  # to Skip tensor if it contains any 0 values

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
        
        violations, warnings=(heatmap.heatmap(cor,warningData,violationData))
        violationsVsFrame.append(violations)
        warningsVsFrame.append(warnings)
        vioDict = {'vio': violationsVsFrame}
        warnDict = {'warn':warningsVsFrame}
        df = pd.DataFrame(vioDict)
        df2 =pd.DataFrame(warnDict)
        df.to_csv('file.csv')
        df2.to_csv('file2.csv')


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
