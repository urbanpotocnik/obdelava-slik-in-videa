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
    
    iImage = changeSpatialDomain("enlarge", iImage, m, n)

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

    oImage = changeSpatialDomain("reduce", oImage, m, n)
    return oImage

# Naloga 3:
def changeSpatialDomain(iType, iImage, iX, iY, iMode=None, iBgr=0):
    Y,X = iImage.shape

    if iType == "enlarge":
        if iMode is None:
            oImage = np.zeros((Y+2*iY, X+2*iX))
            oImage[iY:Y+iY, iX:X+iX] = iImage

    elif iType == "reduce":
        if iMode is None:
            oImage = iImage[iY:Y-iY, iX:X-iX]

    else:
        print("\nError: Incorrect iType!\n")
        return 0        
    
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
    PaddedImage = changeSpatialDomain("enlarge", I, 30, 30)
    displayImage(PaddedImage,"Razširjena sliak z vrednostjo 0")

    # Dodatno: Naloga 2
    filter_5x7 = weightedAverageFilter(7, 5, 2)
    print("Nenormaliziran filter velikosti 5x7 z iValue=2:\n", filter_5x7)
    # Filter, za katerega izberemo iValue = 1, se imenuje "Sobelov filter"