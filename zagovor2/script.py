import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage, transformImage, getRadialValues, getParameters, getParametersUpgraded, changeSpatialDomain, spatialFiltering, thresholdImage

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
    displayImage(Timage2, "Premik 1")

    T2 = getParameters("affine", rot=-angle)
    Timage1 = transformImage("affine", Timage2, pxDim, np.linalg.inv(T2), iBgr=bgr, iInterp=1)
    displayImage(Timage1, "Rotacija")

    # T(450, 70)
    point_t = np.array([450, 70])
    offset2 = imCenter - point_t

    T3 = getParameters("affine", trans=[offset2[0], offset2[1]])
    Timage3 = transformImage("affine", Timage1, pxDim, np.linalg.inv(T3), iBgr=bgr, iInterp=1)
    displayImage(Timage3, "Premik 2")

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

    # Kombiniranje slik v eno
    combinedSobelImage = np.sqrt(sobelImageX**2 + sobelImageY**2)
    combinedSobelImage = (combinedSobelImage / np.max(combinedSobelImage)) * 255  

    displayImage(combinedSobelImage, 'Kombiniran Sobel')

    # TO DO: poslji asistentu, da preveri
    # TO DO: pojdi cez in porihtaj OSVlib






    


