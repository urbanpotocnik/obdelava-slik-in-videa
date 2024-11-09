import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import displayImage

# Naloga 1 -> funkcija za nalaganje 3D slike
def loadImage3D(iPath, iSize, iType):
    fid = open(iPath, 'rb')
    im_shape = (iSize[1], iSize[0], iSize[2]) # Y, X, Z
    oImage = np.ndarray(shape=im_shape, dtype=iType, buffer=fid.read(), order="F")
    fid.close()

    return oImage

# Naloga 2 
# Funkcija za displayanje 2D odseka 3D slike
def displayImage(iImage, iTitle='', iGridX=None, iGridY=None):
    fig = plt.figure()
    plt.title(iTitle)

    if iGridX is not None and iGridY is not None:
        stepX = iGridX[1] - iGridX[0]
        stepY = iGridY[1] - iGridY[0]

        extent = {
            iGridX[0] - 0.5 * stepX,
            iGridX[-1] + 0.5 * stepX,
            iGridY[-1] + 0.5 * stepY,
            iGridY[0] - 0.5 * stepY,
        }

    extent = {
        0 - 0.5,
        iImage.shape[1] - 0.5, 
        iImage.shape[0] - 0.5, 
        0 - 0.5,
    }
    
    # Izračunaj extent, če sta iGridX in iGridY definirana
    if iGridX is not None and iGridY is not None:
        extent = [iGridX[0], iGridX[-1], iGridY[-1], iGridY[0]]  # [xmin, xmax, ymin, ymax]
        plt.imshow(iImage, cmap='gray', vmin=0, vmax=255, aspect='equal', extent=extent)
    else:
        plt.imshow(iImage, cmap='gray', vmin=0, vmax=255, aspect='equal')
    
    plt.show()
    return fig

# Funkcija za dolocanje poljubnega pravokotnega ravninskega prereza
def getPlanarCrossSection(iImage, iDim, iNormVec, iLoc):
    Y, X, Z = iImage.shape
    dx, dy, dz = iDim

    # Stranska ravnina
    if iNormVec == [1, 0, 0]:
        oCS = iImage[:, iLoc, :].T
        oH = np.arange(Y) * dy
        oV = np.arange(Z) * dz
    
    # Celna ravnina
    if iNormVec == [0, 1, 0]:
        oCS = iImage[iLoc, :, :].T
        oH = np.arange(X) * dx
        oV = np.arange(Z) * dz

    # Precna ravnina
    if iNormVec == [0, 0, 1]:
        oCS = iImage[:, :, iLoc]
        oH = np.arange(X) * dx
        oV = np.arange(Y) * dy

    return np.array(oCS), oH, oV

if __name__ == "__main__":
    imSize = [512, 58, 907]
    pxDim = [0.597656, 3, 0.597656]
    I = loadImage3D(r"vaja4/data/spine-512x058x907-08bit.raw", imSize, np.uint8)
    #print(I.shape)
    displayImage(I[:, 250, :], "Prerez")
    
    xc = 290
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [1, 0, 0], xc)
    displayImage(sagCS, "Sagital crosssection", sagH, sagV)

    xc = 30
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [0, 1, 0], xc)
    displayImage(sagCS, "Coronal crosssection", sagH, sagV)

    xc = 500
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [0, 0, 1], xc)
    displayImage(sagCS, "Axial crosssection", sagH, sagV)

# Naloga 3 -> poljubna pravokotna ravninska projekcija
def getPlanarProjection(iImage, iDim, iNormVec, iFunc):
    Y, X , Z = iImage.shape
    dx, dy, dz = iDim

    # Stranska ravnina
    if iNormVec == [1, 0, 0]:
        oP = iFunc(iImage, axis = 1).T
        oH = np.arange(Y) * dy
        oV = np.arange(Z) * dz

    # Celna ravnina
    if iNormVec == [0, 1, 0]:
        oP = iFunc(iImage, axis = 0).T
        oH = np.arange(X) * dx
        oV = np.arange(Z) * dz

    # Precna ravnina
    if iNormVec == [0, 0, 1]:
        oP = iFunc(iImage, axis = 2)
        oH = np.arange(X) * dx
        oV = np.arange(Y) * dy

    return oP, oH, oV

if __name__ == "__main__":
    func = np.max
    
    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [1, 0, 0], func)
    displayImage(sagP, "Sagital projection", sagH, sagV)

    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0, 1, 0], func)
    displayImage(sagP, "Coronal projection", sagH, sagV)

    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0, 0, 1], func)
    displayImage(sagP, "Axial projection", sagH, sagV)

# Dodatno: Naloga 1
if __name__ == "__main__":
    imSize = [512, 58, 907]
    pxDim = [0.597656, 3, 0.597656]
    I = loadImage3D(r"vaja4/data/spine-512x058x907-08bit.raw", imSize, np.uint8)
    
    xc = 256
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [1, 0, 0], xc)
    displayImage(sagCS, "Dodatno naloga 1: Sagital crosssection", sagH, sagV)

    yc = 35
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [0, 1, 0], yc)
    displayImage(sagCS, "Dodatno naloga 1: Coronal crosssection", sagH, sagV)

    zc = 467
    sagCS, sagH, sagV = getPlanarCrossSection(I, pxDim, [0, 0, 1], zc)
    displayImage(sagCS, "Dodatno naloga 1: Axial crosssection", sagH, sagV)

# Dodatno: Naloga 2
if __name__ == "__main__":
    func = np.mean
    
    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [1, 0, 0], func)
    displayImage(sagP, "Dodatno naloga 2: Sagital projection", sagH, sagV)

    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0, 1, 0], func)
    displayImage(sagP, "Dodatno naloga 2: Coronal projection", sagH, sagV)

    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0, 0, 1], func)
    displayImage(sagP, "Dodatno naloga 2: Axial projection", sagH, sagV)

# Dodatno: Naloga 3
"""
Pri CT slikah človeškega telesa so najpogosteje uporabljene projekcije maksimalne vrednosti (za kosti), povprečne vrednosti 
(za mehka tkiva) in medianske vrednosti (za zmanjšanje šuma). Projekciji standardnega odklona in variance sta specifični in 
se uporabljata le v raziskovalne namene, medtem ko je minimalna projekcija redko smiselna.
"""

# TO DO: Dodatno naloga 4 in 5