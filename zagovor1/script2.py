import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage

# Naloga 1:
if __name__ == "__main__":
    image = loadImage('zagovor1/data/rose-366-366-08bit.raw', (366, 366), np.uint8)
    figure = displayImage(image, 'Roza')


def getBoundaryIndices(iImage, iAxis):
    Y, X = iImage.shape
    vrstice = []
    stolpci = []

    if iAxis == 1:
        for y in range(Y):
            for x in range(X):
                if iImage[y, x] != 255:
                    stolpci.append(x)
        if stolpci:
            oIdx1 = min(stolpci)
            oIdx2 = max(stolpci)
            return oIdx1, oIdx2

    if iAxis == 2:
        for y in range(Y):
            if np.any(iImage[y, :] != 255):
                vrstice.append(y)
        if vrstice:
            oIdx1 = min(vrstice)
            oIdx2 = max(vrstice)
            return oIdx1, oIdx2

if __name__ == "__main__":
    image = loadImage('zagovor1/data/rose-366-366-08bit.raw', (366, 366), np.uint8)
    oIdx1, oIdx2 = getBoundaryIndices(image, 1)
    print(oIdx1, oIdx2)
    oIdy1, oIdy2 = getBoundaryIndices(image, 2)
    print(oIdy1, oIdy2)

    cropped_image = image[oIdy1:oIdy2, oIdx1:oIdx2]
    cropped = displayImage(cropped_image, 'Obrezana roza')

    # x 24-192
    # y 34-244


# Naloga 2:
def getBoundaryIndices2(iImage):  
    Y, X = iImage.shape

    steblo = []
    listki = []

    for x in range(X):
        if iImage[244, x] != 255:
            steblo.append(x)

    center_x = int(np.mean(steblo))
    S = [center_x, 244]

    Yc = 192 - 24
    Yc_height = int(Yc / 4) + 24

    for x in range(iImage.shape[1]):
        if iImage[Yc_height, x] != 255:
            listki.append(x)

    L = [min(listki), Yc_height]
    D = [max(listki), Yc_height]

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

if __name__ == "__main__":
    S, L, D, fi = getBoundaryIndices2(image)
    print(S, L, D, fi)


# Naloga 3:
def expandImage(iImage):
    Yc, Xc = iImage.shape

    # Nova slika z belim ozadjem
    oImage = np.full((2 * Yc, 2 * Yc), 255, dtype=np.uint8)
    
    # Postavimo sliko v sredino na vrh
    oImage[:Yc, Yc//2:Yc//2 + Xc] = iImage
    
    return oImage


if __name__ == "__main__":
    expanded_image = expandImage(cropped_image)
    displayImage(expanded_image, 'Povečana prostorska domena')