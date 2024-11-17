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
    print(f"I:\tmin={wImage.min()}, max={wImage.max()}")

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
    print(f"I:\tmin={ssImage.min()}, max={ssImage.max()}")

# 5. Naloga
def gammaImage(iImage , gama):
    oImage = np.array(iImage, dtype=float)
    oImage = 255 ** (1 - gama) * (iImage ** gama)

    return oImage

if __name__ == "__main__":
    gImmage = gammaImage(wImage, 5)
    displayImage(gImmage, "Slika po gama preslikavi")
    print(f"I:\tmin={gImmage.min()}, max={gImmage.max()}")

# Dodatno: Naloga 2
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

if __name__ == "__main__":
    tImmage = thresholdImage(wImage, 127)
    displayImage(tImmage, "Slika po upragovanju")

# Dodatno: Naloga 3
def thresholdCurve(iImage):
    dynamic_range = range(int(iImage.min()), int(iImage.max()) + 1)
    pixel_counts = []

    for t in dynamic_range:
        # Štejemo število pikslov ki imajo isto ali manjso vrednost kot prag t
        count = np.sum(iImage <= t)
        pixel_counts.append(count)

    return dynamic_range, pixel_counts

if __name__ == "__main__":
    t_values, counts = thresholdCurve(wImage)

    # Izris grafa
    plt.figure()
    plt.plot(t_values, counts, label="Število pikslov s sg = 0")
    plt.xlabel("Prag t")
    plt.ylabel("Število slikovnih elementov (sg = 0)")
    plt.title("Pragovna funkcija")
    plt.legend()
    plt.grid()
    plt.show()

# Dodatno: Naloga 4
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

if __name__ == "__main__":
    control_points = np.array([[0, 0], [40, 255], [80, 80], [127, 20], [167, 167], [207, 240], [255, 255]])
    iS = control_points[:, 0]
    oS = control_points[:, 1]

    nlImage = nonLinearSectionalScaleImage(wImage, iS, oS)
    displayImage(nlImage, "Odsekoma nelinearna preslikava")
