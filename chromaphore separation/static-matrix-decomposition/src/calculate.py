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
    melanin = []
    hemoglobin = []
    for y in image:
        mrow = []
        hrow = []
        for x in y:
            pixel_vec = np.array([int(x[1])-int(x[2]),
                                int(x[0])-int(x[2]),
                                int(x[0])-int(x[1])])
            decomp = np.dot(decomp_mat, pixel_vec)
            mrow.append(np.abs(decomp[0]).astype(np.uint8))
            hrow.append(np.abs(decomp[1]).astype(np.uint8))
        melanin.append(mrow)
        hemoglobin.append(hrow)
    melanin = np.array(melanin)
    hemoglobin = np.array(hemoglobin)
    return (melanin, hemoglobin)

def melanin_score(melanin_image):
    return np.mean(melanin_image)

def hemoglobin_score(hemoglobin_image):
    return np.mean(hemoglobin_image)