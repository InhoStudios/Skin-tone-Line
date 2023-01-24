import cv2
import numpy as np
from numpy.linalg import norm
import specularity as spc
from model_decomp import decompose
from bilateral_filter import apply_iterative_bilateral_filter, log_transform
from os.path import join
from os import listdir

def remove_specular_from_image(image_path, radius=12, inpaint_method = cv2.INPAINT_NS):
    img = cv2.imread(image_path)
    gray_img = spc.derive_graym(image_path)
    r_img = m_img = np.array(gray_img)

    rimg = spc.derive_m(img, r_img)
    s_img = spc.derive_saturation(img, rimg)
    spec_mask = spc.check_pixel_specularity(rimg, s_img)
    enlarged_spec = spc.enlarge_specularity(spec_mask)
    
    ret_img = cv2.inpaint(img, enlarged_spec, radius, inpaint_method)

    return ret_img, spec_mask

if __name__ == "__main__":
    dir = "final_decomp_images"
    for file in listdir(dir):
        try: 
            print("=======FILE ", file, "=======")
            # preprocessing
            # remove specularity
            spec_removed_img, spec_mask = remove_specular_from_image(join(dir, file))
            print("=======SPECULARITY REMOVED=======")
            # apply log transform
            log_img = log_transform(spec_removed_img)
            # obtain detail layer
            I_dt, I_bs = apply_iterative_bilateral_filter(log_img)
            print("=======FILTER APPLIED=======")

            # decompose into components
            (m_img, h_img) = decompose(I_dt)
            print("=======IMAGE DECOMPOSED=======")
            
            m_file = file.split('.')[0] + "_melanin." + file.split('.')[1]
            h_file = file.split('.')[0] + "_hemoglobin." + file.split('.')[1]
            spec_removed_file = file.split('.')[0] + "_spec_removed." + file.split('.')[1]
            mask_file = file.split('.')[0] + "_spec_mask." + file.split('.')[1]
            log_img_file = file.split('.')[0] + "_log." + file.split('.')[1]
            detail_file = file.split('.')[0] + "_detail." + file.split('.')[1]

            cv2.imwrite(join(dir, "output", m_file), m_img)
            cv2.imwrite(join(dir, "output", h_file), h_img)
            cv2.imwrite(join(dir, "output", spec_removed_file), spec_removed_img)
            cv2.imwrite(join(dir, "output", mask_file), spec_mask)
            cv2.imwrite(join(dir, "output", log_img_file), log_img)
            cv2.imwrite(join(dir, "output", detail_file), I_dt)
            print("=======IMAGES WRITTEN=======\n\n\n")
        except:
            print("error occured... skipping......\n\n\n")
