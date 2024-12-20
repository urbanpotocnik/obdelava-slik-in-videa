import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage

# Naloga 1:
if __name__ == "__main__":
    image = loadImage("vaja10/data/pattern-236x330-08bit.raw", [236,330], np.uint8)
    displayImage(image,"Originalna slika")

def computeDFT(iMatrix, inverse=False):
    N, M = iMatrix.shape
    n = np.arange(N).reshape(1, -1)
    m = np.arange(M).reshape(1, -1)

    WN = 1 / np.sqrt(N) * np.exp(-1j * 2 * np.pi / N) ** (n.T @ n)
    WM = 1 / np.sqrt(M) * np.exp(-1j * 2 * np.pi / M) ** (m.T @ m)

    if inverse:
        WN = np.conj(WN)
        WM = np.conj(WM)

    oMatrix = WN @ iMatrix @ WM
    return oMatrix

if __name__ == "__main__":
    freq_image = computeDFT(image)
    displayImage(freq_image.real, "Slika spektra")

    reconstructed_image = computeDFT(freq_image, inverse=True)
    displayImage(reconstructed_image.real, "Rekonstruirana slika")
    # rekonstruirana slika mora priti nazaj ista 


# Naloga 2:
def analyzeDFT(iMatrix, iOperations, iTitle=""):
    oMatrix = np.array(iMatrix)

    for operation in iOperations:
        if operation == "amplitude":
            oMatrix = np.abs(oMatrix)
        elif operation == "phase":
            oMatrix = np.unwrap(np.angle(oMatrix))
        elif operation == "ln":
            oMatrix = np.log(oMatrix + 1e-10)
        elif operation == "log":
            oMatrix = np.log10(oMatrix + 1e-10)
        elif operation == "scale":
            oMatrix -= oMatrix.min()
            oMatrix /= oMatrix.max()
            oMatrix *= 255
            oMatrix = oMatrix.astype(np.uint8)
        elif operation == "center":
            N, M = oMatrix.shape
            n_c, m_c = int((N - 1) / 2), int((M - 1) / 2)
            A = oMatrix[:n_c, :m_c]
            B = oMatrix[n_c:, :m_c]
            C = oMatrix[n_c:, m_c:]
            D = oMatrix[:n_c, m_c:]

            upper = np.hstack((C, B))
            lower = np.hstack((D, A))

            oMatrix = np.vstack((upper, lower))

        elif operation == "display":
            plt.figure()    
            plt.imshow(oMatrix, aspect="equal", cmap=plt.cm.gray)
            plt.title(iTitle)
            plt.show()

        else:
            raise NotImplementedError(f"Operation {operation} is not implemented!")
        
    return oMatrix
    

if __name__ == "__main__":
    analyzeDFT(freq_image, ["amplitude", "center", "log", "scale", "display"], "Amplitudni spekter")
    analyzeDFT(freq_image, ["phase", "scale", "display"], "Fazni spekter")

# Naloga 3:
def getFilterSpectrum(iMatrix, iD0, iType):
    oMatrix = np.zeros_like(iMatrix, dtype=float)

    N, M = iMatrix.shape

    n_c, m_c = int((N - 1) / 2), int((M - 1) / 2)

    # Idealni nizkopasovni filter IPLF
    if iType[0] == "I":
        for n in range(N):
            for m in range(M):
                D = np.sqrt((m - m_c) ** 2 + (n - n_c) ** 2)
                if D <= iD0:
                    oMatrix[n, m] = 1

    # iType = IPLF, iType[0] = I, iType[1:] = LPF
    if iType[1:] == "HPF":
        oMatrix = 1 - oMatrix

    return oMatrix

if __name__ == "__main__":
    H = getFilterSpectrum(freq_image, min(freq_image.shape) / 10, "ILPF")
    analyzeDFT(H, ["scale", "display"], "ILPF")
    filtered_freq_image = freq_image * analyzeDFT(H, ["center"])
    reconstructed_freq_image = computeDFT(filtered_freq_image, inverse=True)
    analyzeDFT(reconstructed_freq_image, ["amplitude", "display"], "Filtrirana slika")

