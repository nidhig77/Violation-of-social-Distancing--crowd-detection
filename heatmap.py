import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import numpy as np

def heatmap(anum2, warnThresh, violationThresh):
    plt.ion()
    colors = [(20/255, 252/255, 3/255), (252/255, 186/255, 3/255), (1, 0, 0)]  # Green, Yellow, Red
    cmap = LinearSegmentedColormap.from_list('CustomMap', colors)

    plt.clf()
    hm = np.zeros([720, 1280], dtype=float)

    violations = 0
    warnings = 0

    indices = np.where(anum2 == 1)
    xy_pair = list(zip(indices[0], indices[1]))
    for i, j in xy_pair:
        rl = np.clip(i - warnThresh, 0, None)
        ru = np.clip(i + warnThresh, None, 719)
        cl = np.clip(j - warnThresh, 0, None)
        cu = np.clip(j + warnThresh, None, 1279)
        asub = anum2[rl:ru, cl:cu]
        unique, counts = np.unique(asub, return_counts=True)
        num = dict(zip(unique, counts))
        num.setdefault(1, 0)
        if num[1] > 1:
            warnings += num[1]
            hm[rl:ru, cl:cu] = 1
        rl = np.clip(i - violationThresh, 0, None)
        ru = np.clip(i + violationThresh, None, 719)
        cl = np.clip(j - violationThresh, 0, None)
        cu = np.clip(j + violationThresh, None, 1279)
        asub = anum2[rl:ru, cl:cu]
        unique, counts = np.unique(asub, return_counts=True)
        num = dict(zip(unique, counts))
        num.setdefault(1, 0)
        if num[1] > 1:
            violations += num[1]
            hm[rl:ru, cl:cu] = 2

    print("Number of violations:", violations)
    print("Number of warnings:", warnings)

    hm_blurred = cv2.GaussianBlur(hm.astype(float), (499, 499), 0)
    hm_blurred[-1][-1] = 2
    hm_blurred[-1][-2] = 1
    
    plt.imshow(hm_blurred, cmap=cmap, interpolation='gaussian')
    cbar = plt.colorbar(ticks=[0, 1, 2], label='Value')
    cbar.set_ticklabels(['Safe', 'Warning', 'Violation'])
    
    plt.text(0.5, -0.1, f'Number of violations: {violations}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='red')
    plt.text(0.5, -0.15, f'Number of warnings: {warnings}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='orange')
    
    plt.draw()
    return violations, warnings
