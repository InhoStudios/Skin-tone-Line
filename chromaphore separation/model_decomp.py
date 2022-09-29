import numpy as np
import cv2

decomp_mat = np.array([[0.176, 0.747, 1.042], [0.086, 0.352, -0.488]])

img = cv2.imread('109_1.jpg')
# channels = np.array([img[:,:,1]-img[:,:,2],
#                     img[:,:,0]-img[:,:,2],
#                     img[:,:,0]-img[:,:,1]])

melanin = []
hemoglobin = []
for y in img:
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
cv2.imshow("melanin", melanin)
cv2.imshow("hemoglobin", hemoglobin)
cv2.waitKey(0)
# print(channels)
# result = np.dot(decomp_array, channels)