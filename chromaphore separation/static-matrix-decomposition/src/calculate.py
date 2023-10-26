import numpy as np
import cv2

def static_matrix_decomposition(image):
    """
    Applies static matrix decomposition to an image and obtains hemoglobin and melanin component images

    Parameters
    -----
    image: ndarray BGR Image

    Returns
    -----
    (melanin, hemoglobin): Tuple containing a greyscale melanin and hemoglobin image
    """
    decomp_mat = np.array([[0.176, 0.747, 1.042], [0.086, 0.352, -0.488]])

    diff_blue_green = image[:, :, 1] - image[:, :, 2]
    diff_blue_red = image[:, :, 0] - image[:, :, 2]
    diff_green_red = image[:, :, 0] - image[:, :, 1]
    pixel_vec = np.stack((diff_blue_green, diff_blue_red, diff_green_red), axis=-1)

    decomp = np.dot(pixel_vec, decomp_mat.T)

    melanin = np.abs(decomp[:, :, 0])
    hemoglobin = np.abs(decomp[:, :, 1])

    return melanin, hemoglobin

def masked_index_score(image, mask):
    if (len(mask.shape) > len(image.shape)):
        mask = mask[:,:,0]
    targets = image[np.where(mask == np.max(mask))]
    return np.mean(targets)

def melanin_score(melanin_image):
    return np.mean(melanin_image)

def hemoglobin_score(hemoglobin_image):
    return np.mean(hemoglobin_image)