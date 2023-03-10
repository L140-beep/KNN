import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from skimage.measure import regionprops, label
from enum import Enum, auto

class FLAG(Enum):
    DEFAULT = auto(),
    TWO_PART = auto(),
    NDIM2 = auto()


text_images = [plt.imread(path)
               for path in sorted(pathlib.Path("out").glob("*.png"))]

print(len(text_images))

two_part_letters = ['i']

train_images = {}

for path in tqdm(sorted(pathlib.Path("train").glob("*"))):
    symbol = path.name[-1]
    train_images[symbol] = []
    for image_path in path.glob("*.png"):
        train_images[symbol].append(plt.imread(image_path))
    
def extract_features(image, flag=FLAG.DEFAULT):
    match flag:
        case FLAG.DEFAULT:
            gray = np.mean(image, 2)
            gray[gray > 0] = 1
            labeled = label(gray)
        case FLAG.TWO_PART:
            labeled = np.mean(image, 2)
            labeled[labeled > 0] = 1
            labeled = labeled.astype('uint8')
        case FLAG.NDIM2:
            labeled = image.astype("uint8")
    
    props = regionprops(labeled)[0]
    extent = props.extent
    eccentricity = props.eccentricity
    euler = props.euler_number
    
    rr, cc = props.centroid_local
    
    rr = rr / props.image.shape[0]
    cc = cc / props.image.shape[1]
    
    feret = (props.feret_diameter_max - 1) / np.max(props.image.shape)
    
    return np.array([extent, eccentricity, euler, rr, cc, feret]).astype("f4")

knn = cv2.ml.KNearest_create()

train = []
responses = []

classes = {symbol: i for i, symbol in enumerate(train_images)}
class2sym = {value: key for key, value in classes.items()}

for i, symbol in tqdm(enumerate(train_images)):
    flag = FLAG.DEFAULT
    if symbol in two_part_letters:
        flag = FLAG.TWO_PART
    for image in train_images[symbol]:
        train.append(extract_features(image, flag))
        responses.append(classes[symbol])

train = np.array(train).astype("f4")
responses = np.array(responses).reshape(-1, 1).astype("f4")

knn.train(train, cv2.ml.ROW_SAMPLE, responses)

def isTwoPartLetter(prev_region, current_region) -> bool:
    diff = current_region.centroid[1] - prev_region.centroid[1]
    return diff < 10

def concatenateRegions(prev_region, region, source):
    min_y1, min_x1, max_y1, max_x1 = prev_region.bbox

    min_y2, min_x2, max_y2, max_x2 = region.bbox

    max_y = max(max_y1, max_y2)
    min_y = min(min_y1, min_y2)

    max_x = max(max_x1, max_x2)
    min_x = min(min_x1, min_x2)
    
    return source[min_y:max_y, min_x:max_x]

def image_to_text(image) -> str:
    gray = np.mean(image, 2)
    gray[gray > 0] = 1
    labeled = label(gray)
    regions = regionprops(labeled)
    regions = sorted(regions, key= lambda region : region.centroid[1])
    answer = []
    
    prev_region = -1
    
    i = 0
    
    while i < len(regions):
        region = regions[min(i, len(regions) - 1)]
        next_region = regions[min(i + 1, len(regions) - 1)]
        
        img = region.image
        
        if prev_region != -1:
            if isTwoPartLetter(prev_region, region):
                img = concatenateRegions(prev_region, region, gray)

        if isTwoPartLetter(region, next_region):
            img = concatenateRegions(region, next_region, gray)
            i += 1
        
        
        features = extract_features(img, FLAG.NDIM2).reshape(1, -1)
        ret, results, neighbours, dist = knn.findNearest(features, 5)
        answer.append(class2sym[int(ret)])
        prev_region = region
        i += 1
                
        
    # for region in regions:
    #     img = region.image
    #     if prev_region != -1:
    #         if isTwoPartLetter(prev_region, region):
    #             img = concatenateRegions(prev_region, region, gray)
    #         else:
    #             img
    #     if isTwoPartLetter(prev_region, region):
    #         continue    
    #     features = extract_features(img, FLAG.NDIM2).reshape(1, -1)
    #     ret, results, neighbours, dist = knn.findNearest(features, 5)
    #     answer.append(class2sym[int(ret)])
    #     prev_region = region
    return "".join(answer)
    


for image in text_images:
    print(image_to_text(image))