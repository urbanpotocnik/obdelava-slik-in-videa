import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage, changeSpatialDomain

# Naloga 1:
def distancePoint2Line(iL, iP):
    k, n = iL
    xp, yp = iP

    a = -k
    b = 1
    c = -n

    oD = abs(a * xp + b * yp + c) / np.sqrt(a**2 + b**2)
    return oD


if __name__ == "__main__":
    K = 0.22
    N = 100
     
    L = [K, N]
    P = [0, 0]

    distance = distancePoint2Line(L, P)
    print(f"Razdalja med tocko P in premico L: {distance}")


# Naloga 2:
def weightedGaussianFilter(iS, iWR, iStdR, iW):
    M, N = iS
    w1, w2 = iWR
    sigma1, sigma2 = iStdR
    w = iW

    # Linearno interpoliraj standardni odklon glede na utež
    sigma = sigma1 + (sigma2 - sigma1) * (w - w1) / (w2 - w1)

    # Inicializiraj matriko jedra filtra
    oK = np.zeros((N, M))

    # Izračunaj koordinate središča filtra
    centerX = M // 2
    centerY = N // 2

    # Izračunaj vrednosti Gaussove porazdelitve za vsako točko v filtru
    for y in range(N):
        for x in range(M):
            dx = x - centerX
            dy = y - centerY
            oK[y, x] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    # Normaliziraj jedro filtra
    oK /= (2 * np.pi * sigma**2)
    oK /= oK.sum()

    oStd = sigma
    return oK, oStd


if __name__ == "__main__":
    M = 7
    N = 7
    w1 = 0
    w2 = 10
    sigma1 = 0.1
    sigma2 = 10
    w = 5 

    iS = [M, N]
    iWR = [w1, w2]
    iStdR = [sigma1, sigma2]
    iW = w

    oK, oStd = weightedGaussianFilter(iS, iWR, iStdR, iW)
    print("Jedro filtra:")
    print(oK)
    print(f"Standardni odklon: {oStd}")


# Naloga 3:
def applyGaussianFilter(iImage, kernel):
    M, N = iImage.shape
    m, n = kernel.shape
    pad_y, pad_x = m // 2, n // 2

    padded_image = np.pad(iImage, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
    oImage = np.zeros_like(iImage)

    for y in range(M):
        for x in range(N):
            region = padded_image[y:y + m, x:x + n]
            oImage[y, x] = np.sum(region * kernel)

    return oImage

def imitateMiniature ( iImage , iS , iStdR , iL , iD0 ) :

    Y, X = iImage.shape
    d1 = distancePoint2Line(iL, [0,0])
    d2 = distancePoint2Line(iL, [0,Y-1])
    d3 = distancePoint2Line(iL, [X-1,Y-1])
    d4 = distancePoint2Line(iL, [X-1, 0])

    d_max = max(d1,d2,d3,d4) 
    M, N = iS 
    m = int((M-1)/2) 
    n = int((N-1)/2)

    iImage = changeSpatialDomain('enlarge', iImage, iX = m, iY = n) 

    Y, X = iImage.shape 

    oImage = np.array(iImage, dtype = float)
    oVal = []

    for y in range (n, Y-n):
        for x in range(m, X-m):
            d = distancePoint2Line(iL, [x,y]) 
            if d > iD0:
                K, std = weightedGaussianFilter(iS, [iD0, d_max], iStdR, d)
                patch = iImage[y-n:y+n+1, x-m:x+m+1]
                oImage[y,x] = np.sum(patch*K)
                oVal.append([d, std]) 

    oImage = changeSpatialDomain('reduce', oImage, iX = m, iY = n)

    return oImage , oVal

if __name__ == "__main__":
    I = loadImage("zagovor6/data/train-400x240-08bit.raw", (400,240), np.uint8) 
    
    displayImage(I)
    iP = [0,100]
    iL = [0.22, 100]
    test = distancePoint2Line(iL, iP)

    wr = [0,10]
    stdr = [0.1, 10]
    sz = [7,7]
    K, std = weightedGaussianFilter(sz, wr, stdr, 10)

    mI, vals = imitateMiniature(I, sz, iStdR = stdr, iL = iL, iD0 = 25)
    displayImage(mI)

    vals = np.array(vals)

    plt.figure()
    plt.plot(vals[:,0], vals[:,1])
    plt.grid()
    plt.show()