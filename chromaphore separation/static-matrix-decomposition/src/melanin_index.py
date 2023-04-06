import pandas as pd

import requests

import numpy as np
import matplotlib.pyplot as plt

from os.path import join, exists
from os import listdir

from specularity_removal import remove_specular_from_image
from bilateral_filter import apply_iterative_bilateral_filter, log_transform
from model_decomp import calculate_melanin_index

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
HEADERS = {'User-Agent': USER_AGENT}
IMAGE_DIRECTORY = "../data/melanin_index"


def download_images(data):
    for row in data.values:
        url = row[np.where(data.columns == 'url')][0]
        host_filename = url.split("/")[-1]
        compiled_filename = join(IMAGE_DIRECTORY, f"{row[0]}{host_filename}")

        if exists(compiled_filename):
            continue

        img_bytes = requests.get(url, headers=HEADERS).content

        with (open(compiled_filename, 'wb') as image):
            image.write(img_bytes)
        break

def process_images(image_dir):
    melanin_vals = []
    for file in listdir(image_dir):
        try:
            img_path = join(image_dir, file)
            print(f"Retriving image at {img_path}")
            spec_removed_img, _ = remove_specular_from_image(img_path)
            print(f"Specular reflection removed, proceeding to log transform for {file}")
            log_img = log_transform(spec_removed_img)
            print(f"Log transform completed, proceeding to filtering for {file}")
            I_dt, I_bs = apply_iterative_bilateral_filter(log_img, diam=10, sigmaColor=80, sigmaSpace=10, maxIterations=1000)
            print(f"Filtering completed, proceeding to calculations for {file}")
            melanin_val = calculate_melanin_index(I_dt)
            melanin_vals.append(melanin_val)
        except:
            print("ERROR")
            melanin_vals.append(0)
    return melanin_vals

def get_fitzpatrick_column(data):
    return data.values[:, 1]

def main():
    csv_path = "../data/fitzpatrick17k.csv"
    data = pd.read_csv(csv_path)
    download_images(data)
    fitzpatrick_arr = get_fitzpatrick_column(data)
    melanin_arr = process_images(IMAGE_DIRECTORY)
    plt.scatter(fitzpatrick_arr[0:len(melanin_arr)], melanin_arr)
    plt.show()


if __name__ == "__main__":
    main()
