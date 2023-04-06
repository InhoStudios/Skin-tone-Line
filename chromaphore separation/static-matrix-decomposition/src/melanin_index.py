import pandas as pd

import requests

import numpy as np
import cv2

from os.path import join, exists
from os import listdir

from specularity_removal import remove_specular_from_image

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
HEADERS = {'User-Agent': USER_AGENT}
IMAGE_DIRECTORY = "../data/melanin_index"


def download_images(csv_path):
    data = pd.read_csv(csv_path)
    print(data.columns)
    for row in data.values:
        url = row[np.where(data.columns == 'url')][0]
        host_filename = url.split("/")[-1]
        compiled_filename = join(IMAGE_DIRECTORY, row[0] + host_filename)

        if exists(compiled_filename):
            continue

        img_bytes = requests.get(url, headers=HEADERS).content

        with (open(compiled_filename, 'wb') as image):
            image.write(img_bytes)

def process_images(image_dir):
    fitzpatrick_types = []
    melanin_indices = []
    for file in listdir(image_dir):
        try:
            spec_removed_img, spec_mask = remove_specular_from_image(file)
        except:
            pass

def main():
    csv_path = "../data/fitzpatrick17k.csv"
    download_images(csv_path)


if __name__ == "__main__":
    main()
