from cv2 import cvtColor
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
from os import listdir
from os.path import join

def segment(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(img, lower_threshold, upper_threshold)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    skin = cv2.bitwise_and(img, img, mask=mask)

    return cv2.cvtColor(skin, cv2.COLOR_HSV2RGB)

def remove_black(estimator_labels, estimator_cluster):
    has_black = False
    occurance_counter = Counter(estimator_labels)

    def compare(x, y): return Counter(x) == Counter(y)

    for x in occurance_counter.most_common(len(estimator_cluster)):
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        if compare(color, [0, 0, 0]):
            del occurance_counter[x[0]]
            has_black = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break
    
    return (occurance_counter, estimator_cluster, has_black)

def get_color_information(estimator_labels, estimator_cluster, thresholding=False):
    occurance_counter = None
    color_info = []

    has_black = False

    if thresholding:
        (occurance, cluster, black) = remove_black(estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        has_black = black
    else:
        occurance_counter = Counter(estimator_labels)
    
    total_occurance = sum(occurance_counter.values())

    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        index = (index - 1) if ((thresholding & has_black) & (int(index) != 0)) else index

        color = estimator_cluster[index].tolist()

        color_percentage = (x[1]/total_occurance)

        info = {
            "cluster_index": index,
            "color": color,
            "color_percentage": color_percentage
        }

        color_info.append(info)
    
    return color_info

def extract_dominant_color(image, colors=5, thresholding=False):
    if thresholding:
        colors += 1

    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    estimator = KMeans(n_clusters=colors, random_state=0)

    estimator.fit(img)

    color_info = get_color_information(estimator.labels_, estimator.cluster_centers_, thresholding)

    return color_info

def color_bar(color_info):
    color_bar = np.zeros((100, 500, 3), dtype='uint8')

    top_x = 0
    for x in color_info:
        bottom_x = top_x + (x['color_percentage'] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)

        top_x = bottom_x
    
    return color_bar

def get_avg_hue(img_list):
    hues = []
    for image in img_list:
        colors = color_bar(extract_dominant_color(segment(image), colors=10, thresholding=True))
        print('obtained colour')
        hsv_colors = cvtColor(colors, cv2.COLOR_BGR2HSV)
        point_avg = np.mean(hsv_colors[:, :, 0].flatten())
        hues.append(point_avg)
    return np.mean(hues)

def hue_as_angle(hue): return (hue/255) * 360

path = "skin segmentation/test-images"
imgs = []
for img in listdir(path):
    full_path = join(path, img)
    imgs.append(cv2.imread(full_path))

hue = get_avg_hue(imgs)
print(hue_as_angle(hue))