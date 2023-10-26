import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import cv2
import preprocessing as p
from colour_constancy import shades_of_grey as sog
import calculate as calc
import matplotlib.pyplot as plt

from datetime import datetime

import matplotlib
matplotlib.use('Agg')

from os.path import join, exists
from os import listdir, makedirs

PARENT_DIR = "/scratch/st-tklee-1/ndsz/stl/"

DATADIR = "../data/PIH_dataset/images"
LESION_MASK_DIR = "../data/PIH_dataset/lesion_masks"
NORMAL_MASK_DIR = "../data/PIH_dataset/normal_masks"
METADATA_DIR = "../data/PIH_dataset/PIH_CM_IS_Fiber_Demographics_Lab_Melanin.xlsx"

def to_image_space(image):
    # ret_image = image.copy() - np.min(image) / (np.max(image) - np.min(image))
    return (255 * image.copy()).astype(np.uint8)

if __name__ == "__main__":
    sigmaSpace = 60
    sigmaColor = 15
    maxIter = 1000
    diam = 6 * sigmaSpace
    atol = 0.05

    now = datetime.now()
    dateCode = now.strftime("%Y%m%d")[2:]
    
    parameterized_folder = f"S_SPACE_{sigmaSpace}_S_COL_{sigmaColor}_{maxIter}_Iter_{atol}_tol"
    WORKDIR = join(PARENT_DIR, parameterized_folder)

    makedirs(WORKDIR, exist_ok=True)
    mel_normals = []
    mel_lesions = []
    calc_normals = []
    calc_lesions = []

    dfs = pd.read_excel(METADATA_DIR, sheet_name=None)
    melanin_gt = dfs['IS-Melanin']

    for file in sorted(listdir(LESION_MASK_DIR)):
        if not file.startswith("PIH"):
            continue
        print(file)
        case = file.split(".")[0]
        mel_idx = melanin_gt.index.values[melanin_gt.Patient == case][0]

        mel_les = melanin_gt._get_value(mel_idx, "Unnamed: 3")
        mel_norm = melanin_gt._get_value(mel_idx, "Unnamed: 7")

        mel_lesions.append(mel_les)
        mel_normals.append(mel_norm)
        
        def_file = join(WORKDIR, file)
        spec_file = join(WORKDIR, f"spec_removed_{file}")
        cc_file = join(WORKDIR, f"cc_{file}") 
        log_file = join(WORKDIR, f"log_{file}")
        dt_file = join(WORKDIR, f"detail_{file}")
        bs_file = join(WORKDIR, f"baseline_{file}")
        m_file = join(WORKDIR, f"melanin_{file}")
        h_file = join(WORKDIR, f"hemoglobin_{file}")

        if file == "Thumbs.db":
            continue

        img_file = join(DATADIR, file)
        les_mask = cv2.imread(join(LESION_MASK_DIR, file))
        norm_mask = cv2.imread(join(NORMAL_MASK_DIR, file))

        print(img_file)

        if exists(m_file):
            continue
        
        cv2.imwrite(join(WORKDIR, file), cv2.imread(img_file))
        image, mask = p.remove_specular_from_img(img_file, radius=15)
        image = sog(image, 6)
        cv2.imwrite(cc_file, image)
        print(img_file)

        im = p.normalize(image)
        im = p.neg_log_transform(im)
        cv2.imwrite(log_file, (255 * im).astype(int))

        im_dt, im_bs = p.apply_iterative_bilateral_filter(im, diam=diam, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace, maxIterations=maxIter)
        print(im_dt.dtype)

        cv2.imwrite(bs_file, to_image_space(im_bs))
        cv2.imwrite(dt_file, to_image_space(im_dt))
        im_m, im_h = calc.static_matrix_decomposition(im_dt)
        
        calc_les = calc.masked_index_score(im_m, les_mask)
        calc_norm = calc.masked_index_score(im_m, norm_mask)
        calc_lesions.append(calc_les)
        calc_normals.append(calc_norm)

        cv2.imwrite(m_file, to_image_space(im_m))
        cv2.imwrite(h_file, 10 * to_image_space(im_h))

        csv_file = join(WORKDIR, f"{parameterized_folder}.csv")
        correlation_file = join(WORKDIR, f"{parameterized_folder}_correlations.csv")

        data = np.asarray([mel_les, mel_norm, calc_les, calc_norm]).T

        if (exists(csv_file)):
            prev_data = np.loadtxt(csv_file, delimiter=",").astype(float)
            data = np.vstack((prev_data, data))
        
        np.savetxt(csv_file, data, delimiter=",")

        try:
            columnwise = data.copy().T

            les_corr = pearsonr(columnwise[0], columnwise[2])
            norm_corr = pearsonr(columnwise[1], columnwise[3])

            corrs = [les_corr[0], les_corr[1], norm_corr[0], norm_corr[1]]

            np.savetxt(correlation_file, corrs, delimiter=",")
        except:
            pass

        fig, axs = plt.subplots(2, 2, figsize=(20,20))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")
        axs[0, 1].imshow(im)
        axs[0, 1].set_title("Negative Log")
        axs[0, 1].axis("off")
        axs[1, 0].imshow(im_m, cmap="gray")
        axs[1, 0].set_title("Melanin Image")
        axs[1, 0].axis("off")
        axs[1, 1].imshow(im_h, cmap="gray")
        axs[1, 1].set_title("Hemoglobin Image")
        axs[1, 1].axis("off")

        fig.tight_layout()
        plt.savefig(join(WORKDIR, f"comparison_{file}"))
        plt.close()

    fig, axs = plt.subplots(2, 1, figsize=(20,20))
    axs[0].scatter(calc_lesions, mel_lesions)
    axs[0].set_title("Lesion skin melanin values")
    axs[1].scatter(calc_normals, mel_normals)
    axs[1].set_title("Normal skin melanin values")
    fig.tight_layout()
    plt.savefig(join(WORKDIR, f"correlation_scatter.png"))
    plt.close()