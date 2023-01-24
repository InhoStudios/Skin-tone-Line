import numpy as np
import cv2
from os.path import join
from os import listdir

decomp_mat = np.array([[0.176, 0.747, 1.042], [0.086, 0.352, -0.488]])

# channels = np.array([img[:,:,1]-img[:,:,2],
#                     img[:,:,0]-img[:,:,2],
#                     img[:,:,0]-img[:,:,1]])

def decompose(image):
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

def process_images(path):
    images = []
    for file in listdir(path):
        img = cv2.imread(join(path,file))
        if img is not None:
            (m, h) = decompose(img)
            m_file = file.split('.')[0] + "_m." + file.split('.')[1]
            h_file = file.split('.')[0] + "_h." + file.split('.')[1]
            cv2.imwrite(join(path, "/melanin", m_file), m)
            cv2.imwrite(join(path, "/hemoglobin", h_file), h)

if __name__ == "__main__":
    path = "final_decomp_images"
    process_images(path)
