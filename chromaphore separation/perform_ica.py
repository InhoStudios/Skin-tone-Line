from sklearn.decomposition import FastICA
from skimage import data, io, color

import cv2

img_signal = io.imread("109_1.jpeg", as_grey=True)
# to obtain channel: img_signal[:,:,channel]

ica = FastICA(n_components=2)
s_ = ica.fit_transform(img_signal)

print(s_)
