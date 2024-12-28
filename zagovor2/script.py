import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import displayImage, transformImage, getParameters, spatialFiltering, thresholdImage

# Naloga 1:
# slika je 807x421
def color2grayscale(iImage):
    iImage = iImage * 255
    oImage = iImage.mean(axis=2).astype(int)

    return oImage

if __name__ == "__main__":
    iImage = plt.imread('zagovor2/data/paris_map-807-421.png')
    oImage = color2grayscale(iImage)
    displayImage(oImage,"Originalna slika")

# Naloga 2:
# A = [352, 155], B = [400, 171] 
if __name__ == "__main__":
    point_a = np.array([352, 155])
    point_b = np.array([400, 171])
    image_vector = point_b - point_a
    unit_vector = np.array([1, 0])
    phi = 180 - np.arccos((unit_vector @ image_vector) / np.linalg.norm(image_vector)) * 180 / np.pi
    print(phi)

    angle = 180 - phi
    print(angle)

    imSize = [807, 421]
    imCenter = [403.5 , 210.5]
    offset = imCenter - point_a
    pxDim = [1, 1]
    bgr = 0
    
    T1 = getParameters("affine", trans=[offset[0], offset[1]])
    Timage2 = transformImage("affine", oImage, pxDim, np.linalg.inv(T1), iBgr=bgr, iInterp=1)
    displayImage(Timage2, "Translacija 1")

    T2 = getParameters("affine", rot=-angle)
    Timage1 = transformImage("affine", Timage2, pxDim, np.linalg.inv(T2), iBgr=bgr, iInterp=1)
    displayImage(Timage1, "Rotacija")

    # T(450, 70)
    point_t = np.array([450, 70])
    offset2 = imCenter - point_t

    T3 = getParameters("affine", trans=[offset2[0], offset2[1]])
    Timage3 = transformImage("affine", Timage1, pxDim, np.linalg.inv(T3), iBgr=bgr, iInterp=1)
    displayImage(Timage3, "Translacija 2")

    # NOTE: Tu se mi je najlazje zdelo resitev izvesti na tak nacin, zanima me ce je resitev prevec "nastrikana"


# Naloga 3:
if __name__ == "__main__":
    SobelX = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])
    Sobely = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])
    sobelImageX = spatialFiltering("kernel", Timage3, iFilter=SobelX)
    sobelImageY = spatialFiltering("kernel", Timage3, iFilter=Sobely)
    sobelImageX = thresholdImage(sobelImageX, 250)
    sobelImageY = thresholdImage(sobelImageY, 250)

    displayImage(sobelImageX, 'Sobel X')
    displayImage(sobelImageY, 'Sobel Y')

    # Kombiniranje filtra v osi x in v osi y v isto sliko
    combinedSobelImage = np.sqrt(sobelImageX**2 + sobelImageY**2)
    combinedSobelImage = (combinedSobelImage / np.max(combinedSobelImage)) * 255  

    displayImage(combinedSobelImage, 'Kombiniran Sobel')


def getCenterPoint(iImage, sideLength):
    Y, X = iImage.shape
    oAcc = np.zeros((Y, X))

    halfSide = sideLength / 2
    rangeF = np.arange(0, 360, 1)
    rangeFrad = np.deg2rad(rangeF)
    idxF = np.arange(len(rangeF))

    for y in range(Y):
        for x in range(X):
            if iImage[y, x]:
                for f_idx in idxF:
                    fi = rangeFrad[f_idx]
                    x0 = int(x - halfSide * np.cos(fi))
                    y0 = int(y - halfSide * np.sin(fi))

                    if 0 <= x0 < X and 0 <= y0 < Y:
                        oAcc[y0, x0] += 1

    max_value = np.max(oAcc)
    center_points = np.argwhere(oAcc == max_value)
    oCenter = center_points[0] if len(center_points) > 0 else None

    print(f"Center point: {oCenter}")
    return oCenter, oAcc

if __name__ == "__main__":
    center_point, oAcc = getCenterPoint(combinedSobelImage, 53)
    print(oAcc)

    plt.imshow(Timage3, cmap='gray')
    plt.scatter(center_point[1], center_point[0], c='red', s=20) 
    plt.title('Center')
    plt.show()

# TO DO: v OSVlib dodaj color2grayscale in kombiniranje sobelovega filtra

