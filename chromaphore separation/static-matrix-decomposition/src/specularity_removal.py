import cv2
import numpy as np
import specularity as spc

def remove_specular_from_image(image_path, radius=12, inpaint_method = cv2.INPAINT_NS):
    img = cv2.imread(image_path)
    gray_img = spc.derive_graym(image_path)
    r_img = m_img = np.array(gray_img)

    rimg = spc.derive_m(img, r_img)
    s_img = spc.derive_saturation(img, rimg)
    spec_mask = spc.check_pixel_specularity(rimg, s_img)
    enlarged_spec = spc.enlarge_specularity(spec_mask)
    
    ret_img = cv2.inpaint(img, enlarged_spec, radius, inpaint_method)

    return ret_img

# impath = 'sample_image.jpg'
# img = cv2.imread(impath)
# gray_img = spc.derive_graym(impath)
#
# r_img = m_img = np.array(gray_img)
#
# rimg = spc.derive_m(img, r_img)
# s_img = spc.derive_saturation(img, rimg)
# spec_mask = spc.check_pixel_specularity(rimg, s_img)
# enlarged_spec = spc.enlarge_specularity(spec_mask)
#
# radius = 12
# ns = cv2.inpaint(img, enlarged_spec, radius, cv2.INPAINT_NS)

# cv2.imshow("original", img)
# cv2.imshow("specular mask", spec_mask)
# cv2.imwrite("spec_mask.jpg", spec_mask)
# cv2.imshow("inpainting with navier-stokes", ns)
# cv2.imwrite("inpaint_ns.jpg", ns)

# cv2.waitKey(0)