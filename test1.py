import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label

# tire = plt.imread('train/-/0.png')
img = plt.imread('out/1.png')

gray = np.mean(img, 2)
gray[gray > 0] = 1

gray = gray.astype('uint8')

labeled = label(gray)

regions = regionprops(labeled)
regions = sorted(regions, key= lambda region : region.centroid[1])

for i in range(len(regions)):
    region = regions[i]
    next_region = regions[min(i + 1, len(regions) - 1)]
    min_y1, min_x1, max_y1, max_x1 = region.bbox
    min_y2, min_x2, max_y2, max_x2 = next_region.bbox
    
    diff = min_x2 - max_x1
    
    print(max_x1, min_x2, diff)
    if diff > 45:
        print("Space")

plt.imshow(labeled)
plt.show()
