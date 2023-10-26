import numpy as np
import pandas as pd
import cv2
import preprocessing as p
from colour_constancy import shades_of_grey as sog
import calculate as calc
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

from os.path import join, isdir
from os import listdir, makedirs

WORKDIR = "/scratch/st-tklee-1/ndsz/stl/mit"
DATADIR = "../data/melanin_index_test"
LOGFILE = join(WORKDIR, "meta.log")

if __name__ == "__main__":
    log = ""
    makedirs(WORKDIR, exist_ok=True)
    is_mels = []
    fib_mels = []
    calc_vals = []

    for img in sorted(listdir(DATADIR)):
        dt_file = join(WORKDIR, f"detail_{img}")
        cc_file = join(WORKDIR, f"cc_{img}")
        log_file = join(WORKDIR, f"log_{img}")
        m_file = join(WORKDIR, f"melanin_{img}")
        h_file = join(WORKDIR, f"hemoglobin_{img}")

        if (not img.endswith(".jpg")):
            continue
        img_file = join(DATADIR, img)
        # image = cv2.imread(img_file)

        image, mask = p.remove_specular_from_img(img_file, radius=15)
        image = sog(image, 6)
        cv2.imwrite(cc_file, image)
        print(img_file)

        im = p.log_transform(image)
        cv2.imwrite(log_file, im)
        im = p.normalize(im)

        im_dt, im_bs = p.apply_iterative_bilateral_filter(im, diam=10, sigmaColor=80, sigmaSpace=10, maxIterations=1000)
        # im_dt, im_bs = p.hsv_shading_removal(im)
        cv2.imwrite(dt_file, im_dt)
        im_m, im_h = calc.static_matrix_decomposition(im_dt)
        
        m_index = calc.melanin_score(im_m)
        
        log = log + f"{img}: {m_index}\n"

        cv2.imwrite(m_file, im_m)
        cv2.imwrite(h_file, 25 * im_h)

        fig, axs = plt.subplots(2, 2, figsize=(20,20))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")
        axs[0, 1].imshow(im)
        axs[0, 1].set_title("Log and No-spec")
        axs[0, 1].axis("off")
        axs[1, 0].imshow(im_m, cmap="gray")
        axs[1, 0].set_title("Melanin Image")
        axs[1, 0].axis("off")
        axs[1, 1].imshow(im_h, cmap="gray")
        axs[1, 1].set_title("Hemoglobin Image")
        axs[1, 1].axis("off")

        fig.tight_layout()
        plt.savefig(join(WORKDIR, f"comparison_{img}"))

        with open(LOGFILE, 'w') as f:
            f.write(log)
    
    # plt.scatter(calc_vals, is_mels, label="Calculated Melanin Values")
    # plt.scatter(calc_vals, fib_mels, label="Fiber Melanin Values")
    # plt.legend()
    # plt.savefig(join(WORKDIR, "correlations.png"))
