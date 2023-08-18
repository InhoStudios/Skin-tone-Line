import numpy as np
import pandas as pd
import cv2
import preprocessing as p
from colour_constancy import shades_of_grey as sog
import calculate as calc
import matplotlib.pyplot as plt
from os.path import join
from os import listdir, makedirs

WORKDIR = "/scratch/st-tklee-1/ndsz/stl/test4"
DATADIR = "../data/PIH Study - Image Analysis"
METADATA_DIR = "../data/PIH Study - Image Analysis/PIH_CM_IS_Fiber_Demographics_Lab_Melanin.xlsx"

if __name__ == "__main__":
    makedirs(WORKDIR, exist_ok=True)
    is_mels = []
    fib_mels = []
    calc_vals = []

    dfs = pd.read_excel(METADATA_DIR, sheet_name=None)
    is_melanin = dfs['IS-Melanin']
    fiber_melanin = dfs['Fiber-Melanin']

    for file in sorted(listdir(DATADIR)):
        if not file.startswith("PIH"):
            continue
        print(file)
        img_dir = join(DATADIR, file)
        is_idx = is_melanin.index.values[is_melanin.Patient == file][0]
        fib_idx = fiber_melanin.index.values[fiber_melanin.Patient == file][0]
        is_mel = is_melanin._get_value(is_idx, "Unnamed: 7")
        fib_mel = fiber_melanin._get_value(fib_idx, "Unnamed: 7")
        print(is_mel, fib_mel)
        
        for img in listdir(img_dir):
            dt_file = join(WORKDIR, f"detail_{img}")
            cc_file = join(WORKDIR, f"cc_{img}")
            log_file = join(WORKDIR, f"log_{img}")
            m_file = join(WORKDIR, f"melanin_{img}")
            h_file = join(WORKDIR, f"hemoglobin_{img}")

            if img == "Thumbs.db":
                continue
            img_file = join(img_dir, img)
            image = cv2.imread(img_file)

            # image = p.remove_specular_from_img(img_file, radius=15)
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

            is_mels.append(is_mel)
            fib_mels.append(fib_mel)
            calc_vals.append(m_index)

            cv2.imwrite(m_file, im_m)
            cv2.imwrite(h_file, 100 * im_h)

            data = np.asarray([calc_vals, is_mels, fib_mels])
            np.savetxt(join(WORKDIR, "meta.csv"), data, delimiter=",")

            fig, axs = plt.subplots(2, 2, figsize=(20,20))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axs[0, 0].imshow(image)
            axs[0, 0].set_title("Original Image")
            axs[0, 0].axis("off")
            axs[0, 1].imshow(im)
            axs[0, 1].set_title("Log and No-spec")
            axs[0, 1].axis("off")
            axs[1, 0].imshow(im_m)
            axs[1, 0].set_title("Melanin Image")
            axs[1, 0].axis("off")
            axs[1, 1].imshow(im_h)
            axs[1, 1].set_title("Hemoglobin Image")
            axs[1, 1].axis("off")

            fig.tight_layout()
            plt.savefig(join(WORKDIR, f"comparison_{img}"))
            break
        break
            
    # plt.scatter(calc_vals, is_mels, label="Calculated Melanin Values")
    # plt.scatter(calc_vals, fib_mels, label="Fiber Melanin Values")
    # plt.legend()
    # plt.savefig(join(WORKDIR, "correlations.png"))
