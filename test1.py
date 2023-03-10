import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label

# tire = plt.imread('train/-/0.png')
img = plt.imread('train/si/0.png')

gray = np.mean(img, 2)
gray[gray > 0] = 1

gray = gray.astype('uint8')

labeled = label(gray)

regions = regionprops(labeled)
regions = sorted(regions, key= lambda region : region.centroid[1])
prev_region = regions[0]
region = regions[1]

min_y1, min_x1, max_y1, max_x1 = prev_region.bbox

min_y2, min_x2, max_y2, max_x2 = region.bbox

max_y = max(max_y1, max_y2)
min_y = min(min_y1, min_y2)

max_x = max(max_x1, max_x2)
min_x = min(min_x1, min_x2)

new_image = labeled[min_y:max_y, min_x:max_x]



plt.imshow(new_image)
plt.show()
