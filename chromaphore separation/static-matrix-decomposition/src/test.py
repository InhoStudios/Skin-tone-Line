# from model_decomp import DecompositionMatrix, decompose
import cv2
from colour_constancy import shades_of_grey
import matplotlib.pyplot as plt
# import numpy as np
# from runner import remove_specular_from_image
# from bilateral_filter import apply_iterative_bilateral_filter, log_transform

# decomp_mat = DecompositionMatrix()

# fname = "../data/sample_lesion.jpeg"
# im = cv2.imread(fname)
# spec_removed_im, spec_mask = remove_specular_from_image(fname)
# log_img = log_transform(spec_removed_im)
# I_dt, I_bs = apply_iterative_bilateral_filter(spec_removed_im, diam=15, sigmaColor=80, sigmaSpace=10, maxIterations=2000, fname="process.avi")
# (m_img, h_img) = decomp_mat.decompose(I_dt)
        
# m_file = "../data/custom_mat_melanin.jpeg"
# h_file = "../data/custom_mat_hemoglobin.jpeg"

# cv2.imwrite(m_file, m_img)
# cv2.imwrite(h_file, h_img)
print("Running")

img_pth = "../data/final_decomp_images/DSC_0007.jpg"

image = cv2.imread(img_pth)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.savefig("/scratch/st-tklee-1/ndsz/stl/original.png")

plt.close()

grey_im = shades_of_grey(image, 6)

plt.figure(figsize=(10, 10))
plt.imshow(grey_im)
plt.savefig("/scratch/st-tklee-1/ndsz/stl/cc_applied.png")
# plt.show()