import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import numpy

def heatmap(anum2):
    plt.ion()
    colors = [(20/255, 252/255, 3/255), (252/255, 186/255, 3/255), (1, 0, 0)]  # Green, Yellow, Red
    cmap = LinearSegmentedColormap.from_list('CustomMap', colors)

    plt.clf()
    hm = numpy.zeros([720,1280], dtype=float)

    violations = 0

    indices = numpy.where(anum2==1)
    xy_pair = list(zip(indices[0],indices[1]))
    for i,j in xy_pair:
        rl = numpy.clip(i - 120, 0, None)
        ru = numpy.clip(i + 120, None, 719)
        cl = numpy.clip(j - 120, 0, None)
        cu = numpy.clip(j + 120, None, 1219)
        asub = anum2[rl:ru, cl:cu]
        unique, counts = numpy.unique(asub, return_counts=True)
        num = dict(zip(unique, counts))
        num.setdefault(1, 0)
        if num[1]>1:
            hm[rl:ru, cl:cu] = 1
        rl = numpy.clip(i - 100, 0, None)
        ru = numpy.clip(i + 100, None, 719)
        cl = numpy.clip(j - 100, 0, None)
        cu = numpy.clip(j + 100, None, 1219)
        asub = anum2[rl:ru, cl:cu]
        unique, counts = numpy.unique(asub, return_counts=True)
        num = dict(zip(unique, counts))
        num.setdefault(1, 0)
        if num[1]>1:
            violations+= num[1]
            hm[rl:ru, cl:cu] = 2

    print("Number of violations:",violations)

    hm_blurred = cv2.GaussianBlur(hm.astype(float), (249, 249), 0)
    
    plt.imshow(hm_blurred, cmap=cmap, interpolation='gaussian')
    plt.colorbar(ticks=[0, 1, 2], label='Value').set_ticklabels(['Safe', 'Warning', 'Violation'])
    plt.draw()
    return(violations)