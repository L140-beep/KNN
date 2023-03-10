import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
image = plt.imread('out/1.png')

gray = np.mean(image, 2)
gray[gray > 0] = 1

# gray = gray.astype('uint8')

labeled = label(gray)

regions = regionprops(labeled)

regions = sorted(regions, key= lambda region : region.centroid[1])

last = regions[0]

# plt.imshow(labeled)
# plt.waitforbuttonpress(0)

for i in regions:
    dif = i.centroid[1] - last.centroid[1]
    print(i.coords[0][1], last.coords[-1][1], i.coords[0][1] - last.coords[-1][1])
    # print(last.centroid[1], i.centroid[1], dif)
    plt.imshow(i.image)
    plt.waitforbuttonpress(0)
    if dif < 10 and dif != 0:
        min_y1, min_x1, max_y1, max_x1 = last.bbox

        min_y2, min_x2, max_y2, max_x2 = i.bbox

        max_y = max(max_y1, max_y2)
        min_y = min(min_y1, min_y2)

        max_x = max(max_x1, max_x2)
        min_x = min(min_x1, min_x2)

        new_image = labeled[min_y:max_y, min_x:max_x] 
        print("TWO-PART LETTER!!")
        plt.imshow(new_image)
        plt.waitforbuttonpress(0)
    
    last = i
plt.imshow(labeled)
plt.show()