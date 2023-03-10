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

def image_to_text(image) -> str:
    gray = np.mean(image, 2)
    gray[gray > 0] = 1
    labeled = label(gray)
    regions = regionprops(labeled)
    regions = sorted(regions, key= lambda region : region.centroid[1])
    answer = []
    
    for region in regions:
        features = extract_features(region.image, FLAG.NDIM2).reshape(1, -1)
        ret, results, neighbours, dist = knn.findNearest(features, 5)
        answer.append(class2sym[int(ret)])
    
    return "".join(answer)
    


for image in text_images:
    print(image_to_text(image))