import matplotlib.pyplot as plt
import numpy as np
import os, sys

def loadImage(iPath, iSize, iType):
    with open(iPath, 'rb') as fid:
        buffer = fid.read()

    buffer_len = len(np.frombuffer(buffer=buffer, dtype=iType))
    
    if buffer_len != np.prod(iSize):
        raise ValueError('Size of the data does not match the specified size')
    else:
        oImage_shape = (iSize[1], iSize[0])

    oImage = np.ndarray(oImage_shape, dtype = iType, buffer = buffer, order = 'F')
    return oImage

def displayImage(iImage, iTitle='', iGridX=None, iGridY=None):
    fig = plt.figure()
    plt.title(iTitle)
    
    # Izračunaj extent, če sta iGridX in iGridY definirana
    if iGridX is not None and iGridY is not None:
        extent = [iGridX[0], iGridX[-1], iGridY[-1], iGridY[0]]  # [xmin, xmax, ymin, ymax]
        plt.imshow(iImage, cmap='gray', vmin=0, vmax=255, aspect='equal', extent=extent)
    else:
        plt.imshow(iImage, cmap='gray', vmin=0, vmax=255, aspect='equal')
    
    plt.show()
    return fig

def saveImage(iImage, iPath, iType):
    with open(iPath, 'wb') as fid:      # 'rb'=read mode, 'wb'=write mode
        fid.write(iImage.tobytes())
        fid.close()                     # there is no need for the close method, but it is written in the manual

def computeHistogram(iImage):
    nBits = int(np.log2(iImage.max())) + 1  # Ta vrstica izračuna število bitov, ki je potrebno za predstavitev najvišje intenzitete v sliki

    oLevels = np.arange(0, 2 ** nBits, 1)   # Ustvarjanje nivojev intenzitete

    iImage = iImage.astype(np.uint8)        # Pretvorba slike v uint8

    oHist = np.zeros(len(oLevels))          # Inicializacija praznega histograma

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            oHist[iImage[y, x]] = oHist[iImage[y, x]] + 1

    # Normaliziran histogram -> prikaže nam porazdelitev vrednosti
    oProb = oHist / iImage.size
    
    # CDF 
    oCDF = np.zeros_like(oHist)
    for i in range(len(oProb)):
        oCDF[i] = oProb[: i + 1].sum()
                   
    return oHist, oProb, oCDF, oLevels

def displayHistogram(iHist, iLevels, iTitle):
    plt.figure()
    plt.title(iTitle)
    plt.bar(iLevels, iHist, width = 1, edgecolor = "darkred", color = "red")
    plt.xlim(iLevels.min(), iLevels.max())
    plt.ylim(0, 1.05 * iHist.max())
    plt.show()

def equalizeHistogram(iImage):
    _, _, CDF, _ = computeHistogram(iImage)    # Izračun CDFja iz podane slike

    nBits = int(np.log2(iImage.max())) + 1
    
    max_intensity = 2 ** nBits + 1

    oImage = np.zeros_like(iImage)              # Ustvari isto sliko kot je iImage, vendar vse piksle nastavi na 0

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):        # Novo intenziteto izračunamo tako, da uporabimo vrednost CDF za staro intenziteto in jo pomnožimo z največjo možno intenziteto max_intensity
            old_intensity = iImage[y, x]
            new_intensity = np.floor(CDF[old_intensity] * max_intensity)
            oImage[y, x] = new_intensity
    
    return oImage

# Entropija slike je mera za količino informacije, ki jo vsebuje slika
def computeEntropy(iImage):
    hist, _, _, _  = computeHistogram(iImage)
    height, width = iImage.shape                      # Izračunanje višine in širine slike
    nPixels = height * width                         
    probabilities = np.zeros_like(hist, dtype=float)  # Inicializacija arraya
    
    probabilities = hist / nPixels

    entropy = 0

    for p in probabilities: 
       if p > 0:                        # Ignoriramo ničlo
            entropy += p * np.log2(p)
    
    oEntropy = -entropy

    return oEntropy

def addNoise(iImage, iStd):
    height, width = iImage.shape
    oNoise = np.random.randn(height, width) * iStd
    iImage = iImage.astype(np.uint8)
    oImage = oNoise + iImage
    
    return oImage, oNoise

def interpolateImage(iImage, iSize, iOrder):
    iOrder = int(iOrder)
    Y, X = iImage.shape

    M, N = iSize

    oImage = np.zeros((N, M), dtype = iImage.dtype)

    dx = (X - 1) / (M - 1)
    dy = (Y - 1) / (N - 1)

    for n in range(N):
        for m in range(M):
            s = 0

            pt = np.array([m * dx, n * dy])

            # 0 red interpolacije
            if iOrder == 0:
                # Najdi najblizjega soseda
                px = np.round(pt).astype(np.uint16)
                s = iImage[px[1], px[0]]

            if iOrder == 1:
                px = np.floor(pt).astype(np.uint16)

                # Racunanje utezi
                a = abs(pt[0] - (px[0] + 1)) * abs(pt[1] - (px[1] + 1))
                b = abs(pt[0] - (px[0] + 0)) * abs(pt[1] - (px[1] + 1))
                c = abs(pt[0] - (px[0] + 1)) * abs(pt[1] - (px[1] + 0))
                d = abs(pt[0] - (px[0] + 0)) * abs(pt[1] - (px[1] + 0))

                # Sivinske 
                sa = iImage[px[1] + 0, px[0] + 0]
                sb = iImage[px[1] + 0, min(px[0] + 1, X - 1)]
                sc = iImage[min(px[1] + 1, Y - 1), px[0] + 0]
                sd = iImage[min(px[1] + 1, Y -1), min(px[0] + 1, X -1)]

                s = int(a * sa + b * sb + c * sc + d * sd)



            oImage[n, m] = s
    
    return oImage

def analyzeInterpolationRegion(image, start_cooridnates, region_size):
    start_x, start_y = start_cooridnates
    region_width, region_height = region_size

    region = image[start_y:start_y + region_height, start_x:start_x + region_width]

    displayImage(region, "Izrezano interpolacijsko območje")
    plt.show()

    hist, prob, CDF, levels = computeHistogram(region)
    displayHistogram(hist, levels, "Histogram interpolacijskega območja")
    plt.show()

    # Izračunaj minimalne, maksimalne in povprečne sivinske vrednosti
    min_val = np.min(region)
    max_val = np.max(region)
    avg_val = np.mean(region)

    print(f"Minimalna sivinska vrednost: {min_val}")
    print(f"Maksimalna sivinska vrednost: {max_val}")
    print(f"Povprečna sivinska vrednost: {avg_val}")

    return min_val, max_val, avg_val

def decimateImage(iImage, iKernel, iLevel):
    # Normaliziramo jedro tako, da vsota elementov postane enaka 1
    iKernel = iKernel / np.sum(iKernel)
    
    oImage = iImage.copy()
    
    # Za vsak nivo decimacije ponovimo postopek konvolucije in decimacije
    for level in range(iLevel):
        image_h, image_w = oImage.shape
        kernel_h, kernel_w = iKernel.shape
        
        # Oblikujemo prazno sliko za shranjevanje rezultatov konvolucije
        pad_h, pad_w = kernel_h // 2, kernel_w // 2
        padded_image = np.pad(oImage, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # Ustvarimo novo prazno sliko za trenutni nivo konvolucije
        convolved_image = np.zeros_like(oImage)
        
        # Konvolucija
        for i in range(image_h):
            for j in range(image_w):
                region = padded_image[i:i + kernel_h, j:j + kernel_w]
                convolved_image[i, j] = np.sum(region * iKernel)
        
        # Decimacija: vzamemo vsak drugi piksel v obeh smereh
        oImage = convolved_image[::2, ::2]
    
    return oImage

def loadImage3D(iPath, iSize, iType):
    fid = open(iPath, 'rb')
    im_shape = (iSize[1], iSize[0], iSize[2]) # Y, X, Z
    oImage = np.ndarray(shape=im_shape, dtype=iType, buffer=fid.read(), order="F")
    fid.close()

    return oImage

# Dodelana funkcija iz 1. vaje
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

# TO DO: zrihti jupyter