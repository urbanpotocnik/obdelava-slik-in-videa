import cv2.version
import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import displayImage
import cv2

# Naloga 1:
def loadFrame(iVideo, iFrame):
    iVideo.set(1, iFrame - 1)
    ret, oFrame = iVideo.read()
    oFrame = oFrame[:, :, 0].astype(float)
    
    return oFrame


if __name__ == "__main__":
    video_path = "vaja11/data/simple-video.avi"
    cap = cv2.VideoCapture(video_path)
    

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Stevilo frameov v videu: {frame_count}")

    N = 100  # Example frame number
    I1 = loadFrame(cap, N)
    I2 = loadFrame(cap, N + 1)
    displayImage(I1, "Frame 100")
    displayImage(I2, "Frame 101")


# Naloga 2:
def framePrediction(iFrame, iMV):
    iMV = np.array(iMV).astype(int)
    dx, dy = iMV

    # Krozno premaknemo vrstice in stolpce
    oFrame = np.roll(iFrame, [dy, dx], axis=(0, 1))

    if dx >= 0:
        oFrame[:, :dx] = -1
    else:
        oFrame[:, dx:] = -1

    if dy >= 0: 
        oFrame[:dy, :] = -1
    else:   
        oFrame[dy:, :] = -1

    return oFrame


def blockMatching(iF1, iF2, iSize, iSearchSize):
    Y , X = iF1.shape
    dx, dy = iSize

    M = int(X / dx) 
    N = int(Y / dy)

    oMF = np.zeros((N, M, 2), dtype=int)
    oCP = np.zeros((N, M, 2), dtype=float)
    Err = np.ones((N, M), dtype=float) * 255

    P = (iSearchSize - 1) / 2
    PTS = np.array([
        [0,0],
        [1,0], [-1,0],
        [0,1], [0,-1],  
        [1,1], [-1,1], [1,-1], [-1,-1]
    ])

    for n in range(N):
        y_min = n * dy
        y_max = (n + 1) * dy
        y = np.arange(y_min, y_max)

        for m in range(M):
            x_min = m * dx
            x_max = (m + 1) * dx
            x = np.arange(x_min, x_max)

            oCP[n, m, 0] = x.mean()
            oCP[n, m, 1] = y.mean()

            # trenuten blok na frameu 2
            B2 = iF2[y_min:y_max, x_min:x_max]

            for i in range(1,4):
                # Logaritemsko skaliranje premika
                P1 = (P + 1) / (2 ** i)
                PTSi = PTS * P1

                # Prvi kandidat za iskanje vektorja premika
                d0 = oMF[n, m, :]

                for p in range(PTSi.shape[0]):
                    # Treutni vektor premika
                    d = d0 + PTSi[p, :]

                    pF2 = framePrediction(iF1, d)
                    pB2 = pF2[y_min:y_max, x_min:x_max]

                    # Maska za odrstranit -1 od prej
                    msk = pB2 >= 0

                    bErr = np.mean(np.abs(B2[msk] - pB2[msk]))

                    if bErr < Err[n, m]:
                        Err[n, m] = bErr
                        oMF[n, m, :] = d
    
    return oMF, oCP 

def displayMotionField(iMF, iCP, iTitle, iImage = None):
    if iImage is None:
        fig = plt.figure()
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal")
        plt.title(iTitle)

    else:
        fig = displayImage(iImage, iTitle)

    plt.quiver(
        iCP[:, :, 0],
        iCP[:, :, 1],
        iMF[:, :, 0],
        iMF[:, :, 1],
        color="r",
        scale=0.5,
        units="xy",
        angles="xy"
    )
    plt.show()

    return fig


if __name__ == "__main__":
    MF, CP = blockMatching(I1, I2, [8, 8], 15)
    fig1 = displayMotionField(MF, CP, "Motion Field")
    fig2 = displayMotionField(MF, CP, "Slika z vektorjem premika", I1)