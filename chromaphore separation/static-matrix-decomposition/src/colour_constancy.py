import numpy as np

def shades_of_grey(input_image, p):

    output_image = input_image

    input_image = np.abs(input_image)

    if (p != -1):
        illum = np.power(input_image, p)
        white_R = np.power(np.sum(np.sum(illum[:, :, 0])), 1/p)
        white_G = np.power(np.sum(np.sum(illum[:, :, 1])), 1/p)
        white_B = np.power(np.sum(np.sum(illum[:, :, 2])), 1/p)

        dist = np.sqrt(white_R ** 2 + white_G ** 2 + white_B ** 2)

    else:
        R = input_image[:, :, 0]
        G = input_image[:, :, 1]
        B = input_image[:, :, 2]

        white_R = np.max(R[:])
        white_G = np.max(G[:])
        white_B = np.max(B[:])

        dist = np.sqrt(white_R ** 2 + white_G ** 2 + white_B ** 2)

    white_R = white_R / dist
    white_G = white_G / dist
    white_B = white_B / dist

    print(dist)
    print((white_R * np.sqrt(3)), (white_G * np.sqrt(3)), (white_B * np.sqrt(3)))

    output_image[:, :, 0] = output_image[:, :, 0] / (white_R * np.sqrt(3))
    output_image[:, :, 1] = output_image[:, :, 1] / (white_G * np.sqrt(3))
    output_image[:, :, 2] = output_image[:, :, 2] / (white_B * np.sqrt(3))

    return output_image
