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

def scaleImage(iImage, a, b):
    oImage = np.array(iImage, dtype=float)
    oImage = a * iImage + b

    return oImage

def windowImage(iImage, iC, iW):
    oImage = np.array(iImage, dtype=float)
    oImage = 255/iW * (iImage - (iC - iW/2))    # Skaliranje vrednosti vhodne slike na skalo 0-255

    oImage[iImage < iC - iW/2] = 0              # Clipnemo sliko na range 0-255
    oImage[iImage > iC + iW/2] = 255

    return oImage

def sectionalScaleImage(iImage, iS, oS):
    oImage = np.array(iImage, dtype=float)

    for i in range(len(iS) - 1):
        sL = iS[i]
        sH = iS[i+1]

        idx = np.logical_and(iImage >= sL, iImage <= sH)
        
        # Scale faktor
        k = (oS[i + 1] - oS[i]) / (sH - sL)

        oImage[idx] = k * (iImage[idx] - sL) + oS[i]

    return oImage

def gammaImage(iImage , gama):
    oImage = np.array(iImage, dtype=float)
    oImage = 255 ** (1 - gama) * (iImage ** gama)

    return oImage

def thresholdImage(iImage, iT):
    Lg = 2 ** 8
    oImage = np.array(iImage, dtype=float)

    for i in range(iImage.shape[0]):
        for j in range(iImage.shape[1]):
            if iImage[i, j] <= iT:
                oImage[i, j] = 0
            else:
                oImage[i, j] = Lg - 1
        
    return oImage

def thresholdCurve(iImage):
    dynamic_range = range(int(iImage.min()), int(iImage.max()) + 1)
    pixel_counts = []

    for t in dynamic_range:
        # Štejemo število pikslov ki imajo isto ali manjso vrednost kot prag t
        count = np.sum(iImage <= t)
        pixel_counts.append(count)

    return dynamic_range, pixel_counts

def nonLinearSectionalScaleImage(iImage, iS, oS):
    oImage = np.zeros_like(iImage, dtype=float)
    
    for i in range(0, len(iS) - 2, 2):  # Obdelujemo po tri točke naenkrat
        sf = iS[i:i+3]
        sg = oS[i:i+3]

        A = np.array([
            [sf[0]**2, sf[0], 1],  
            [sf[1]**2, sf[1], 1],  
            [sf[2]**2, sf[2], 1]  
        ])

        B = np.array(sg)  

        # Izračunamo rešitev sistema
        coefficients = np.linalg.solve(A, B)
        print("Koeficienti A, B, C:", coefficients)
    
        Ai, Bi, Ci = coefficients
        mask = (iImage >= sf[0]) & (iImage <= sf[2])
        oImage[mask] = Ai * iImage[mask]**2 + Bi * iImage[mask] + Ci

    return oImage

def getParameters(iType, scale = None, trans = None, rot = None, shear = None, orig_pts = None, mapped_pts = None):
    # za afino preslikavo: scale je vektor skaliranja oz. povecave, trans je vektor translacije oz. premika, rot = φ je kot rotacije
    # oz. vrtenja v kotnih stopinjah in shear je vektor striga oz. zatega 
    # za radialno preslikavo: orig_pts je matrika kontrolnih tock in mapped_pts je matrika preslikanih kontrolnih tock

    oP = {}

    if iType == "affine":
        if scale is None:
            scale = [1,1]
        if trans is None:
            trans = [0,0]
        if rot is None:
            rot = 0
        if shear is None:
            shear = [0,0]

        Tk = np.array([
            [scale[0],0,0],
            [0,scale[1],0],
            [0,0,1]
        ])

        Tt = np.array([
            [1,0,trans[0]],
            [0,1,trans[1]],
            [0,0,1]
        ])

        phi = rot*np.pi / 180

        Tf = np.array([
            [np.cos(phi),-np.sin(phi),0],
            [np.sin(phi),np.cos(phi),0],
            [0,0,1]
        ])

        Tg = np.array([
            [1,shear[0],0],
            [shear[1],1,0],
            [0,0,1]
        ])        

        oP = Tg @ Tf @ Tt @ Tk

    elif iType == "radial":
        assert orig_pts is not None, "Manjkajo orig_pts"
        assert mapped_pts is not None, "Manjkajo mapped_pts"

        K = orig_pts.shape[0]

        UU = np.zeros((K,K), dtype=float)

        for i in range(K):
            UU[i,:] = getRadialValues(orig_pts[i,:], orig_pts)

        oP["alphas"] = np.linalg.solve(UU, mapped_pts[:,0])
        oP["betas"] = np.linalg.solve(UU, mapped_pts[:,1])
        oP["pts"] = orig_pts
        
    return oP

def getRadialValues(iXY, iCP):
    K = iCP.shape[0]

    oValue = np.zeros(K)

    x_i, y_i = iXY
    for k in range(K):
        x_k, y_k = iCP[k]

        r = np.sqrt((x_i - x_k) ** 2 + (y_i -y_k) ** 2)

        if r > 0:
            oValue[k] = -(r**2) * np.log(r)

    return oValue

def transformImage(iType, iImage, iDim, iP, iBgr=0, iInterp=0):
    Y, X = iImage.shape
    oImage = np.ones((Y, X)) * iBgr
    dx, dy = iDim

    for y in range(Y):
        for x in range(X):
            x_hat, y_hat = x * dx, y * dy

            if iType == "affine":
                x_hat, y_hat, _ = iP @ np.array([x_hat, y_hat, 1])

            if iType == "radial":
                U = getRadialValues([x_hat, y_hat], iP["pts"])
                x_hat, y_hat = np.array([U @ iP["alphas"], U @ iP["betas"]])

            x_hat, y_hat = x_hat / dx, y_hat / dy

            if iInterp == 0:  # interpolacija 0. reda
                x_hat, y_hat = round(x_hat), round(y_hat)

                if 0 <= x_hat < X and 0 <= y_hat < Y:
                    oImage[y, x] = iImage[y_hat, x_hat]

            if iInterp == 1:  # interpolacija 1. reda

                x0 = int(np.floor(x_hat))
                y0 = int(np.floor(y_hat))
                x1 = x0 + 1
                y1 = y0 + 1

                if 0 <= x0 < X and 0 <= y0 < Y and 0 <= x1 < X and 0 <= y1 < Y:
                    a = abs(x_hat - x0) * abs(y_hat - y1)
                    b = abs(x_hat - x1) * abs(y_hat - y1)
                    c = abs(x_hat - x1) * abs(y_hat - y0)
                    d = abs(x_hat - x0) * abs(y_hat - y0)

                    s = (
                        a * iImage[y0, x1]
                        + b * iImage[y0, x0]
                        + c * iImage[y1, x0]
                        + d * iImage[y1, x1]
                    )

                    oImage[y, x] = int(s)
    return oImage

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