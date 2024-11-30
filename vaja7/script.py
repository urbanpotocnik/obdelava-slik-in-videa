import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage

# Naloga 1:
if __name__ == "__main__":
    I = loadImage("vaja7/data/cameraman-256x256-08bit.raw", [256,256], np.uint8)
    displayImage(I,"Originalna slika")

# Naloga 2:
def spatialFiltering(iType, iImage, iFilter, iStatFunc=None, iMorphOp=None):
    N,M = iFilter.shape
    m = int((M-1)/2)
    n = int((N-1)/2)
    
    iImage = changeSpatialDomain("enlarge", iImage, m, n, 0, 0)

    Y,X = iImage.shape
    oImage = np.zeros((Y,X), dtype=float)

    for y in range(n, Y-n):
        for x in range(m,X-m):
            patch = iImage[y-n:y+n+1, x-m:x+m+1]
            
            if iType == "kernel":
                oImage[y,x] = (patch * iFilter).sum()
            elif iType == "statistical":
                oImage[y,x] = iStatFunc(patch)                 
            elif iType == "morphological":
                R = patch[iFilter!=0]
                if iMorphOp == "erosion":
                    oImage[y,x]=R.min()
                elif iMorphOp == "dialation":
                    oImage[y,x]=R.max()
                else:
                    print("\nError: Incorrect iMorphOp!\n")
                    return 0                                 
            else:
                print("\nError: Incorrect iType!\n")
                return 0        

    oImage = changeSpatialDomain("reduce", oImage, m, n, 0, 0)
    return oImage

# Naloga 3:
def changeSpatialDomain(iType, iImage, iX, iY, iMode, iBgr):
    Y,X = iImage.shape

    if iType == "enlarge":
        oImage = np.zeros((Y+2*iY, X+2*iX))
        oImage[iY:Y+iY, iX:X+iX] = iImage

    elif iType == "reduce":
        oImage = iImage[iY:Y-iY, iX:X-iX]

    else:
        print("\nError: Incorrect iType!\n")
        return 0  

    if iMode == "constant":
        oImage = np.zeros((Y+2*iY, X+2*iX)) + iBgr
        oImage[iY:Y+iY, iX:X+iX] = iImage

    elif iMode == "extrapolation":
        oImage = np.zeros((Y+2*iY, X+2*iX)) 
        oImage[iY:Y+iY, iX:X+iX] = iImage

        oImage[:iY, iX:X + iX] = iImage[0, :]
        oImage[Y + iY:, iX:X + iX] = iImage[-1, :]
        oImage[iY:Y + iY, :iX] = iImage[:, 0].reshape(-1, 1)
        oImage[iY:Y + iY, X + iX:] = iImage[:, -1].reshape(-1, 1)
        oImage[:iY, :iX] = iImage[0, 0]
        oImage[:iY, X + iX:] = iImage[0, -1]
        oImage[Y + iY:, :iX] = iImage[-1, 0]
        oImage[Y + iY:, X + iX:] = iImage[-1, -1]


    elif iMode == "reflection":
        oImage = np.zeros((Y + 2 * iY, X + 2 * iX), dtype=iImage.dtype)
        oImage[iY:Y + iY, iX:X + iX] = iImage

        for y in range(iY):
            idx = (iY - y) % Y
            oImage[y, iX:X + iX] = iImage[idx, :]
        for y in range(iY):
            idx = (Y - (y % Y) - 1)
            oImage[Y + iY + y, iX:X + iX] = iImage[idx, :]

        for x in range(iX):
            idx = (iX - x) % X
            oImage[:, x] = oImage[:, iX + idx]
        for x in range(iX):
            idx = (X - (x % X) - 1)
            oImage[:, X + iX + x] = oImage[:, iX + idx]

        for y in range(iY):
            for x in range(iX):
                idx_y_top = (iY - y) % Y
                idx_x_left = (iX - x) % X
                idx_y_bottom = (Y - (y % Y) - 1)
                idx_x_right = (X - (x % X) - 1)

                oImage[y, x] = iImage[idx_y_top, idx_x_left]
                oImage[y, X + iX + x] = iImage[idx_y_top, idx_x_right]
                oImage[Y + iY + y, x] = iImage[idx_y_bottom, idx_x_left]
                oImage[Y + iY + y, X + iX + x] = iImage[idx_y_bottom, idx_x_right]


    elif iMode == 'period':
        if iImage.ndim == 3:
            oImage = np.zeros((Y + 2*iY, X + 2*iX, iImage.shape[2]), dtype=iImage.dtype)
        else:
            oImage = np.zeros((Y + 2*iY, X + 2*iX), dtype=iImage.dtype)

        for y in range(Y):
            for x in range(X):
                oImage[iY + y, iX + x] = iImage[y, x]

        for y in range(iY):
            for x in range(X):
                oImage[y, iX + x] = iImage[(y - iY) % Y, x]

        for y in range(iY):
            for x in range(X):
                oImage[Y + iY + y, iX + x] = iImage[y % Y, x]

        for y in range(Y):
            for x in range(iX):
                oImage[iY + y, x] = iImage[y, (x - iX) % X]

        for y in range(Y):
            for x in range(iX):
                oImage[iY + y, X + iX + x] = iImage[y, x % X]

        for y in range(iY):
            for x in range(iX):
                oImage[y, x] = iImage[(y - iY) % Y, (x - iX) % X]

        for y in range(iY):
            for x in range(iX):
                oImage[y, X + iX + x] = iImage[(y - iY) % Y, x % X]

        for y in range(iY):
            for x in range(iX):
                oImage[Y + iY + y, x] = iImage[y % Y, (x - iX) % X]

        for y in range(iY):
            for x in range(iX):
                oImage[Y + iY + y, X + iX + x] = iImage[y % Y, x % X]

    return oImage            

# Dodatno: Naloga 2
def weightedAverageFilter(iM, iN, iValue):
    oFilter = np.zeros((iN, iM))
    centerY, centerX = iN // 2, iM // 2

    for y in range(iN):
        for x in range(iM):
            distance = abs(centerY - y) + abs(centerX - x)
            oFilter[y, x] = iValue ** (distance)

    return oFilter

if __name__ == "__main__":
    Kernel = np.array([
        [1,1,1],
        [1,-8,1],
        [1,1,1]
    ])
    KImage = spatialFiltering("kernel", I, iFilter=Kernel)
    displayImage(KImage,"Filtrirana slika z Laplaceovim filtrom")
    
    SImage = spatialFiltering("statistical", I, iFilter=np.zeros((30,30)), iStatFunc=np.median)
    displayImage(SImage,"Statisticno filtrirana slika: mediana")

    MKernel = np.array([
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0]
    ])
    MImage = spatialFiltering("morphological", I, iFilter=MKernel, iMorphOp="erosion")
    displayImage(MImage,"Morfološko filtriranje: erozija")

    # Naloga 3:
    PaddedImage = changeSpatialDomain("enlarge", I, 30, 30, 0, 0)
    displayImage(PaddedImage,"Razširjena sliak z vrednostjo 0")

    # Dodatno: Naloga 2
    filter_5x7 = weightedAverageFilter(7, 5, 2)
    print("Nenormaliziran filter velikosti 5x7 z iValue=2:\n", filter_5x7)
    # Filter, za katerega izberemo iValue = 1, se imenuje "Sobelov filter"

    # Dodatno: Naloga 3
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
    sobelImageX = spatialFiltering("kernel", I, iFilter=SobelX)
    sobelImageY = spatialFiltering("kernel", I, iFilter=Sobely)

    amplitude = np.sqrt(sobelImageX**2 + sobelImageY**2)
    phase = np.arctan2(sobelImageY, sobelImageX) * (180 / np.pi)

    amplitude = (amplitude / np.max(amplitude)) * 255
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * 255

    displayImage(sobelImageX, 'Sobel X')
    displayImage(sobelImageY, 'Sobel Y')
    displayImage(amplitude, 'Amplituda')
    displayImage(phase, 'Faza')

    # Dodatno: Naloga 4
    Gauss = np.array([
        [0.01, 0.08, 0.01],
        [0.08, 0.64, 0.08],
        [0.01, 0.08, 0.01]
    ])
    gaussImage = spatialFiltering("kernel", I, iFilter=Gauss)
    
    mask = I - gaussImage
    
    c = 2
    sharpenedImage = I + c * mask

    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask)) * 255
    sharpenedImage = (sharpenedImage - np.min(sharpenedImage)) / (np.max(sharpenedImage) - np.min(sharpenedImage)) * 255
    
    mask = mask.astype(np.uint8)
    sharpenedImage = sharpenedImage.astype(np.uint8)

    displayImage(mask, 'Maska neostrih področij')
    displayImage(sharpenedImage,"Izostrena slika")

    # Dodatno: Naloga 5
    constantImage = changeSpatialDomain("enlarge", I, 128, 384, "constant", 127)
    displayImage(constantImage,"Razširjena sliak z vrednostjo 127")

    extrapolatedImage = changeSpatialDomain("enlarge", I, 128, 384, "extrapolation", 0)
    displayImage(extrapolatedImage,"Ekstrapolirana slika")

    reflectedImage = changeSpatialDomain("enlarge", I, 128, 384, "reflection", 0)
    displayImage(reflectedImage,"Reflektirana slika")

    periodImage = changeSpatialDomain("enlarge", I, 128, 384, "period", 0)
    displayImage(periodImage,"Periodna slika")

    # Dodatno Naloga 6:
    # Način razširitve prostorske domene slike vpliva na rezultate filtriranja, ker različni načini razširitve
    # ustvarjajo različne robne pogoje. Na primer, konstantna vrednost lahko povzroči ostre prehode na robovih,
    # medtem ko periodično ponavljanje ohranja kontinuiteto, kar lahko vodi do bolj gladkih prehodov.
    # Zrcaljenje in ekstrapolacija prav tako vplivata na robne pogoje in s tem na rezultate filtriranja.