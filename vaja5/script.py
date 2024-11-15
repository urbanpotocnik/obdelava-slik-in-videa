import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage, computeHistogram

# 1. Naloga
if __name__ == "__main__":
    I = loadImage("vaja5/data/image-512x512-16bit.raw", [512, 512], np.int16)
    displayImage(I, "Originalna slika")
    print(f"I:\tmin={I.min()}, max={I.max()}")

# 2. Naloga
def scaleImage(iImage, a, b):
    oImage = np.array(iImage, dtype=float)
    oImage = a * iImage + b

    return oImage
    
if __name__ == "__main__":
    sImage = scaleImage(I, -0.125, 256)
    displayImage(sImage, "Slika po skaliranju")
    print(f"I:\tmin={sImage.min()}, max={sImage.max()}")

# 3. Naloga
def windowImage(iImage, iC, iW):
    oImage = np.array(iImage, dtype=float)
    oImage = 255/iW * (iImage - (iC - iW/2))    # Skaliranje vrednosti vhodne slike na skalo 0-255

    oImage[iImage < iC - iW/2] = 0              # Clipnemo sliko na range 0-255
    oImage[iImage > iC + iW/2] = 255

    return oImage

if __name__ == "__main__":
    wImage = windowImage(sImage, 1000, 500)
    displayImage(wImage, "Slika po oknenju")

# 4. Naloga
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

if __name__ == "__main__":
    sCP = np.array([[0, 85], [85, 0], [170, 255], [255, 170]])
    ssImage = sectionalScaleImage(wImage, sCP[:, 0], sCP[:, 1])
    displayImage(ssImage, "Slika po odsekovnem skaliranju") 

# 5. Naloga
def gammaImage(iImage , gama):
    oImage = np.array(iImage, dtype=float)
    oImage = 255 ** (1 - gama) * (iImage ** gama)

    return oImage

if __name__ == "__main__":
    gImmage = gammaImage(wImage, 5)
    displayImage(gImmage, "Slika po gama preslikavi")