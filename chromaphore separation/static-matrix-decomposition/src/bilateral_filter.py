import numpy as np
import cv2
from os.path import join

def log_transform(img_in):
    c = 255.0/(np.log(1 + 255))
    img_out = (c*np.log(1 + img_in.astype(float))).astype(np.uint8)
    return img_out

def antilog_transform(img_in):
    c = c = 255.0/(np.log(1 + 255))
    img_out = (np.exp(img_in.astype(float)) ** (1/c) - 1).astype(np.uint8)
    return img_out

def apply_iterative_bilateral_filter(I_ori, atol=0.05, diam=50, sigmaColor=80, sigmaSpace=80, maxIterations=50, fname="detail."):
    (B_ori, G_ori, R_ori) = cv2.split(I_ori)
    uint8_size = 255

    # convert to floating point with double precision and normalize

    B_ori = B_ori.astype(np.float32) / uint8_size
    G_ori = G_ori.astype(np.float32) / uint8_size
    R_ori = R_ori.astype(np.float32) / uint8_size

    # create diffuse layer copies

    B_d = B_ori.copy()
    G_d = G_ori.copy()
    R_d = R_ori.copy()

    # create empty image as base for detail layer

    B_c = np.zeros(np.shape(B_d)).astype(np.float32)
    G_c = np.zeros(np.shape(G_d)).astype(np.float32)
    R_c = np.zeros(np.shape(R_d)).astype(np.float32)

    # create video writer to watch process

    height, width, dims = I_ori.shape
    detail = cv2.VideoWriter(f'{fname.split(".")[0]}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (width, height))

    for iterator in range(maxIterations):

        B_dp = cv2.bilateralFilter(B_d, diam, sigmaColor/uint8_size, sigmaSpace)
        G_dp = cv2.bilateralFilter(G_d, diam, sigmaColor/uint8_size, sigmaSpace)
        R_dp = cv2.bilateralFilter(R_d, diam, sigmaColor/uint8_size, sigmaSpace)

        B_c = B_c + B_d - B_dp
        G_c = G_c + G_d - G_dp
        R_c = R_c + R_d - R_dp

        B_c[B_c < 0] = 0
        G_c[G_c < 0] = 0
        R_c[R_c < 0] = 0
        
        B_c[B_c > 1] = 1
        G_c[G_c > 1] = 1
        R_c[R_c > 1] = 1

        B_norm = np.max(np.abs(B_d - B_dp))
        G_norm = np.max(np.abs(B_d - B_dp))
        R_norm = np.max(np.abs(B_d - B_dp))

        dispR = R_c.copy()
        dispG = G_c.copy()
        dispB = B_c.copy()

        I_c = cv2.merge([(uint8_size * dispB).astype(np.uint8), 
                         (uint8_size * dispG).astype(np.uint8), 
                         (uint8_size * dispR).astype(np.uint8)])
        detail.write(I_c)

        cv2.imshow("Detail image", I_c)
        cv2.waitKey(1)

        print("Iteration {} -- R: {}, G: {}, B: {}".format(iterator + 1, R_norm, G_norm, B_norm))

        if (B_norm <= atol and G_norm <= atol and R_norm <= atol):
            break
        else:
            B_d = np.abs(B_ori - B_c)
            G_d = np.abs(G_ori - G_c)
            R_d = np.abs(R_ori - R_c)
            iterator += 1

    # convert to uint8 image

    # normalize image
    # scale = np.max([np.max(B_c), np.max(G_c), np.max(R_c)])

    # B_c = B_c / scale
    # G_c = G_c / scale
    # R_c = R_c / scale

    B_c = B_c * uint8_size
    G_c = G_c * uint8_size
    R_c = R_c * uint8_size

    B_c = B_c.astype(np.uint8)
    G_c = G_c.astype(np.uint8)
    R_c = R_c.astype(np.uint8)

    I_dt = cv2.merge([B_c, G_c, R_c])
    I_bs = I_ori - I_dt

    detail.release()

    return I_dt, I_bs

# fname = "../data/sample_lesion.jpeg"
# im = cv2.imread(fname)
# cv2.imshow("Original image", im)
# I_dt, I_bs = apply_iterative_bilateral_filter(im, diam=10, sigmaColor=80, sigmaSpace=3, maxIterations=2000, fname=fname)
# antilog_detail = antilog_transform(I_dt)
# cv2.imshow("Detail image", antilog_detail)
# # cv2.imshow("Baseline image", I_bs)

# cv2.waitKey(0)

# cv2.imwrite("detail.png", I_dt)
# cv2.imwrite("baseline.png", I_bs)

# I_ori = cv2.imread("log_transformed.jpg")[:,:,2]
# cv2.imshow("original", I_ori)
# cv2.waitKey(0)
# I_d = I_ori.copy()

# I_c = np.zeros(np.shape(I_d)).astype(np.uint8)
# for iterator in range(2500):
#     I_dp = cv2.bilateralFilter(I_d, 10, 15, 15)
#     I_c = np.add(I_c.astype(int), np.abs(np.subtract(I_d.astype(int), I_dp.astype(int)))).astype(np.uint8)
#     I_d = np.abs(np.subtract(I_ori.astype(int), I_c.astype(int))).astype(np.uint8)
#     print(iterator)

# I_dt = I_c
# I_bs = np.subtract(I_ori, I_c)

# cv2.imshow("detail", I_dt)
# cv2.imshow("baseline", I_bs)
