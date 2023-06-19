import numpy as np
import cv2
from os.path import join
from os import listdir
from webhelper import get_hemo_data_from_web

decomp_mat = np.array([[0.176, 0.747, 1.042], [0.086, 0.352, -0.488]])

# channels = np.array([img[:,:,1]-img[:,:,2],
#                     img[:,:,0]-img[:,:,2],
#                     img[:,:,0]-img[:,:,1]])

# RED = 612
# GREEN = 549
# BLUE = 465
RED = 650
GREEN = 550
BLUE = 450
# path length estimates
L_R = 10.2e-1
L_G = 4.9e-1
L_B = 0.7e-1
HEM_X = 150 # X given by S. Jacques (1998)

HEMOGLOBIN_EXTCOEFF_URL = "https://omlc.org/spectra/hemoglobin/summary.html"

class DecompositionMatrix:

    def __init__(self):
        self.init_hemoglobin_dict()
        self.create_matrix()
    
    def init_hemoglobin_dict(self):
        hemoglobin_table = get_hemo_data_from_web(HEMOGLOBIN_EXTCOEFF_URL)
        self.hemoglobin_dict = {}
        for row in hemoglobin_table:
            self.hemoglobin_dict[row[0].astype(int)] = row[1:3]
        return

    def create_matrix(self):
        W = np.zeros((3, 2))
        # compose each index of W
        W[0, 0] = self.mu_mel(GREEN) * L_G - self.mu_mel(RED) * L_R
        W[0, 1] = self.mu_hem(GREEN) * L_G
        W[1, 0] = self.mu_mel(BLUE) * L_B - self.mu_mel(RED) * L_R
        W[1, 1] = self.mu_hem(BLUE) * L_B
        W[2, 0] = self.mu_mel(BLUE) * L_B - self.mu_mel(GREEN) * L_G
        W[2, 1] = self.mu_hem(BLUE) * L_B - self.mu_hem(GREEN) * L_G

        self.decomp_mat = np.linalg.inv(W.T @ W) @ W.T
        self.W = W.copy()
        return self.decomp_mat

    def mu_mel(self, lambd):
        return 0.244 + 85.3 * np.exp(-(lambd - 154)/66.2)

    def e(self, lambd):
        assert isinstance(lambd, int)
        # e_arr[0]: oxyhemoglobin
        # e_arr[1]: deoxyhemoglobin
        # hematocrit: 45%
        e_oxy = 0
        e_deoxy = 0
        if (lambd % 2 == 1):
            e_oxy = np.mean([self.hemoglobin_dict[lambd - 1][0], self.hemoglobin_dict[lambd + 1][0]])
            e_deoxy = np.mean([self.hemoglobin_dict[lambd - 1][1], self.hemoglobin_dict[lambd + 1][1]])
            return 0.45 * e_oxy + 0.55 * e_deoxy
        else:
            (e_oxy, e_deoxy) = self.hemoglobin_dict[lambd]
        return 0.45 * e_oxy + 0.55 * e_deoxy
    
    def mu_hem(self, lambd):
        return 2.303 * self.e(lambd) * HEM_X / 64500
    
    def decompose(self, image):
        """
        Applies static matrix decomposition to an image and obtains hemoglobin and melanin component images

        Parameters
        -----
        image: ndarray BGR Image

        Returns
        -----
        (melanin, hemoglobin): Tuple containing a greyscale melanin and hemoglobin image
        """
        melanin = []
        hemoglobin = []
        for y in image:
            mrow = []
            hrow = []
            for x in y:
                pixel_vec = np.array([int(x[1])-int(x[2]),
                                    int(x[0])-int(x[2]),
                                    int(x[0])-int(x[1])])
                decomp = np.dot(self.decomp_mat, pixel_vec)
                mrow.append(np.abs(decomp[0]).astype(np.uint8))
                hrow.append(np.abs(decomp[1]).astype(np.uint8))
            melanin.append(mrow)
            hemoglobin.append(hrow)
        melanin = np.array(melanin)
        hemoglobin = np.array(hemoglobin)
        return (melanin, hemoglobin)
    
    def get_path_lengths(self):
        M = np.zeros((3, 3))
        M[0, 0] = -self.mu_mel(RED)
        M[0, 1] = self.mu_mel(GREEN)
        M[1, 0] = -self.mu_mel(RED)
        M[1, 2] = self.mu_mel(BLUE)
        M[2, 1] = -self.mu_mel(GREEN)
        M[2, 2] = self.mu_mel(BLUE)
        w_vec = self.W[:, 0]
        l_vec = np.linalg.lstsq(M, w_vec, rcond=1)
        print(l_vec)
        return l_vec


def decompose(image):
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

def calculate_melanin_index(image):
    (melanin, hemoglobin) = decompose(image)
    melanin_index = np.mean(melanin)
    return melanin_index

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
    # path = "final_decomp_images"
    # process_images(path)
    path = "../data/melanin_index_test/fair.png"
    im = cv2.imread(path)
    print(calculate_melanin_index(im))
