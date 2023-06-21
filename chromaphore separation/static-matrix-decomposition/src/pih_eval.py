import numpy as np
import pandas as pd
import cv2
import preprocessing as p
import calculate as calc
import matplotlib.pyplot as plt
from os.path import join
from os import listdir, makedirs

WORKDIR = "/scratch/st-tklee-1/ndsz/stl/output"
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
            if img == "Thumbs.db":
                continue
            img_file = join(img_dir, img)
            image = cv2.imread(img_file)
            print(img_file)

            # im = p.remove_specular_from_img(img_file, radius=15)

            dt_file = join(WORKDIR, f"detail_{img}")
            m_file = join(WORKDIR, f"melanin_{img}")
            h_file = join(WORKDIR, f"hemoglobin_{img}")

            im = p.normalize(image)
            im = p.log_transform(im)
            im_dt, im_bs = p.apply_iterative_bilateral_filter(im)
            cv2.imwrite(dt_file, im_dt)
            im_m, im_h = calc.static_matrix_decomposition(im_dt)
            
            m_index = calc.melanin_score(im_m)

            is_mels.append(is_mel)
            fib_mels.append(fib_mel)
            calc_vals.append(m_index)

            cv2.imwrite(m_file, im_m)
            cv2.imwrite(h_file, im_h)

            data = np.asarray([calc_vals, is_mels, fib_mels])
            np.savetxt(join(WORKDIR, "meta.csv"), data, delimiter=",")

            break
    
    plt.plot(calc_vals, label="Calculated Melanin Values")
    plt.plot(is_mels, label="Integrating Sphere Melanin Values")
    plt.plot(fib_mels, label="Fiber Melanin Values")
    plt.legend()
    plt.savefig(join(WORKDIR), "correlations.jpg")
