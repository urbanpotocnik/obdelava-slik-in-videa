import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage, getParameters, transformImage

# Naloga 1:
def getBoundaryIndices(iImage , iAxis):
    iImage = iImage.astype(np.uint8)
    collumns_list = []
    rows_list = []

    # Isci v x smer
    if iAxis == 1:
        for y in range(iImage.shape[0]):
            for x in range(iImage.shape[1]):
                if iImage[y, x] != 255:
                    collumns_list.append(x)

        oIdx1 = min(collumns_list)
        oIdx2 = max(collumns_list)


    # Isci v y smer
    if iAxis == 2:
        for y in range(iImage.shape[0]):
            if np.any(iImage[y] != 255):
                rows_list.append(y)

        oIdx1 = rows_list[0]
        oIdx2 = rows_list[-1]


    return oIdx1 , oIdx2


if __name__ == '__main__':
    image = loadImage('zagovor1/data/rose-366-366-08bit.raw', (366, 366), np.uint8)
    figure = displayImage(image, 'Roza')

    x_1, x_2 = getBoundaryIndices(image, 1)
    print(x_1, x_2)
    y_1, y_2 = getBoundaryIndices(image, 2)
    print(y_1, y_2)

    Xc = x_2 - x_1
    Yc = y_2 - y_1
    print(Xc, Yc)

    cropped_image = image[y_1:y_2, x_1:x_2]
    figure2 = displayImage(cropped_image, 'Obrezana roza')


# Naloga 2:
def getBoundaryIndices2(iImage):
    iImage = iImage.astype(np.uint8)
    height, width = cropped_image.shape
    L_D_height = int(height / 4)

    S_x = []
    S_y = []
    S = []

    L_x = []
    L_y = []
    L = []

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            if iImage[y, x] != 255 and y == height - 1:
                S_x.append(x)
                S_y.append(y)

    S = [int(np.mean(S_x)), int(np.mean(S_y))]

    for x in range(iImage.shape[1]):
        if iImage[L_D_height, x] != 255:
            L_x.append(x)

    L = [min(L_x), L_D_height]
    D = [max(L_x), L_D_height]

    L = np.array(L)
    D = np.array(D)
    S = np.array(S)

    a = L - S
    b = D - S
    fi_rad = np.arccos(
        (a * b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))
    )
    fi = fi_rad * 180 / np.pi

    return S, L, D, fi


if __name__ == '__main__':
    S, L, D, fi = getBoundaryIndices2(cropped_image)
    print(S, L, D, fi)


# Naloga 3:
def expandImage(iImage):
    i_height, i_width = cropped_image.shape
    oImage = 255 * np.ones((2 * i_height, 2 * i_width))

    start_x = int(i_width / 3)
    start_y = 0
    offset_x = int(i_width / 2)

    for start_y in range(i_height):
        for start_x in range(i_width):
            oImage[start_y, start_x + offset_x] = iImage[start_y, start_x]

    return oImage


if __name__ == '__main__':
    expanded_image = expandImage(cropped_image)
    figure3 = displayImage(expanded_image, 'Raztegnjena slika')
    

def createRotatedPattern(iImage, iAngle):
    i_height, i_width = iImage.shape
    oImage = np.zeros_like(iImage, dtype=float)

    imSize = [i_width, i_height]
    pxDim = [1, 1]
    bgr = 255

    center = np.array([i_width - 1, i_height - 1]) / 2
    translation = getParameters("affine", trans=-center)
    translation_inverse = np.linalg.inv(translation)

    angles = np.linspace(0, 360 - iAngle, int(360 / iAngle))
    for angle in angles:
        rotation = getParameters("affine", rot=angle)
        T = translation_inverse @ rotation @ translation
        temp = transformImage("affine", oImage, pxDim, np.linalg.inv(T), bgr, iInterp=0)
        oImage = oImage + temp

    oImage -= oImage.min()
    oImage /= oImage.max()
    oImage *= 255

    return oImage


if __name__ == '__main__':
    rotated_image = createRotatedPattern(cropped_image, 10)
    figure4 = displayImage(rotated_image, 'Rotirana slika')