import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import numpy

def heatmap(anum2,warnThresh,violationThresh):
    
    hm = numpy.zeros([720,1280], dtype=float)

    violations = 0
    warnings = 0

    indices = numpy.where(anum2==1)
    xy_pair = list(zip(indices[0],indices[1]))
    violator=[]
    for i,j in xy_pair:
        z=(j,i)
        rl = numpy.clip(i - warnThresh, 0, None)
        ru = numpy.clip(i + warnThresh, None, 719)
        cl = numpy.clip(j - warnThresh, 0, None)
        cu = numpy.clip(j + warnThresh, None, 1219)
        asub = anum2[rl:ru, cl:cu]
        unique, counts = numpy.unique(asub, return_counts=True)
        num = dict(zip(unique, counts))
        num.setdefault(1, 0)
        if num[1]>1:
            warnings+= num[1]
            hm[rl:ru, cl:cu] = 1
        rl = numpy.clip(i - violationThresh, 0, None)
        ru = numpy.clip(i + violationThresh, None, 719)
        cl = numpy.clip(j - violationThresh, 0, None)
        cu = numpy.clip(j + violationThresh, None, 1219)
        asub = anum2[rl:ru, cl:cu]
        i,j= asub.shape
        unique, counts = numpy.unique(asub, return_counts=True)
        num = dict(zip(unique, counts))
        num.setdefault(1, 0)
        if num[1]>1:
            violations+= num[1]
            hm[rl:ru, cl:cu] = 2
            
            if z not in violator:
                violator.append(z)
        
    print("Number of violations:",violations)
    print("violators cooridinates:" , violator)
    hm_blurred = cv2.GaussianBlur(hm.astype(float), (499, 499), 0)
    hm_blurred[-1][-1]=2
    hm_blurred[-1][-2]=1
    
    
    return(violator, violations,warnings)