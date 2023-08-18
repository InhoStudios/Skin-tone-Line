import numpy as np
from os.path import join
import cv2
import torchvision.transforms as tv
import specularity as spc
from gputools.denoise import bilateral2
# import taichi as ti
# import taichi.math as tm

UINT8_SIZE = 255.0

def normalize(image):
    if not normalized(image):
        return image.astype(np.float32) / UINT8_SIZE
    return image.astype(np.float32)


def normalized(image):
    (b, g, r) = cv2.split(image)
    max_pix = np.max([np.max(b), np.max(g), np.max(r)])
    min_pix = np.min([np.min(b), np.min(g), np.min(r)])
    return max_pix <= 1 and min_pix >= 0

def log_transform(image):
    c = 255.0 / (np.log(1 + 255))
    return (c * np.log(1 + image.astype(float))).astype(np.uint8)

def neg_log_transform(image):
    return -log_transform(image)

def antilog_transform(image):
    return np.exp(image.astype(np.float32))

def convert_to_float32_img(image):
    return image.astype(np.float32)

def remove_specular_from_img(image_path, radius=12, inpaint_method=cv2.INPAINT_NS):
    img = cv2.imread(image_path)
    gray_img = spc.derive_graym(image_path)
    r_img = m_img = np.array(gray_img)

    rimg = spc.derive_m(img, r_img)
    s_img = spc.derive_saturation(img, rimg)
    spec_mask = spc.check_pixel_specularity(rimg, s_img)
    enlarged_spec = spc.enlarge_specularity(spec_mask)
    
    ret_img = cv2.inpaint(img, enlarged_spec, radius, inpaint_method)

    return ret_img, spec_mask


def create_float32_zeros_base(image):
    return np.zeros(image.shape).astype(np.float32)

def hsv_shading_removal(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    flattened_values = cv2.bilateralFilter(hsv_img[:, :, 2], 80, 50, 26)
    mean_v = np.mean(hsv_img[:,:,2])
    print(mean_v)
    hsv_img[:,:,2] = flattened_values
    I_dt = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    I_bs = image - I_dt

    return I_dt, I_bs


def apply_iterative_bilateral_filter(image, atol=0.05, diam=15, sigmaColor=10, sigmaSpace=5, maxIterations=1000):
    # check normalized
    normalize(image)

    (B_ori, G_ori, R_ori) = cv2.split(image)
    
    B_ori, G_ori, R_ori = convert_to_float32_img(B_ori), convert_to_float32_img(G_ori), convert_to_float32_img(R_ori)

    B_d, G_d, R_d = B_ori.copy(), G_ori.copy(), R_ori.copy()

    B_c, G_c, R_c = create_float32_zeros_base(B_ori), create_float32_zeros_base(G_ori), create_float32_zeros_base(R_ori)

    for iterator in range(maxIterations):

        # B_dp = cv2.bilateralFilter(B_d, diam, sigmaColor/UINT8_SIZE, sigmaSpace)
        # G_dp = cv2.bilateralFilter(G_d, diam, sigmaColor/UINT8_SIZE, sigmaSpace)
        # R_dp = cv2.bilateralFilter(R_d, diam, sigmaColor/UINT8_SIZE, sigmaSpace)
        B_dp = bilateral2(B_d, diam, sigmaColor/UINT8_SIZE, sigmaSpace)
        G_dp = bilateral2(G_d, diam, sigmaColor/UINT8_SIZE, sigmaSpace)
        R_dp = bilateral2(R_d, diam, sigmaColor/UINT8_SIZE, sigmaSpace)

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

        # dispR = R_c.copy()
        # dispG = G_c.copy()
        # dispB = B_c.copy()

        # I_c = cv2.merge([(UINT8_SIZE * dispB).astype(np.uint8),
        #                  (UINT8_SIZE * dispG).astype(np.uint8),
        #                  (UINT8_SIZE * dispR).astype(np.uint8)])
        # # detail.write(I_c)

        # cv2.imshow("Detail image", I_c)
        # cv2.waitKey(1)

        if (iterator % 50 == 0):
            print("Iteration {} -- R: {}, G: {}, B: {}".format(iterator, R_norm, G_norm, B_norm))

        if (B_norm <= atol and G_norm <= atol and R_norm <= atol):
            break
        else:
            B_d = np.abs(B_ori - B_c)
            G_d = np.abs(G_ori - G_c)
            R_d = np.abs(R_ori - R_c)
            iterator += 1

    # convert to uint8 image

    B_c = B_c * UINT8_SIZE
    G_c = G_c * UINT8_SIZE
    R_c = R_c * UINT8_SIZE

    B_c = B_c.astype(np.uint8)
    G_c = G_c.astype(np.uint8)
    R_c = R_c.astype(np.uint8)

    I_dt = cv2.merge([B_c, G_c, R_c])
    I_bs = image - I_dt


    return I_dt, I_bs
