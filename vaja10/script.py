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

    q = 2

    n_c, m_c = int((N - 1) / 2), int((M - 1) / 2)

    # Idealni nizkopasovni filter IPLF
    if iType[0:] == "ILPF":
        for n in range(N):
            for m in range(M):
                D = np.sqrt((m - m_c) ** 2 + (n - n_c) ** 2)
                if D <= iD0:
                    oMatrix[n, m] = 1

    # iType = IPLF, iType[0] = I, iType[1:] = LPF
    if iType[0:] == "IHPF":
        oMatrix = 1 - oMatrix

    if iType[0:] == "BLPF":
        for n in range(N):
            for m in range(M):
                D = np.sqrt((m - m_c) ** 2 + (n - n_c) ** 2)
                oMatrix[n, m] = 1 / (1 + (D / iD0) ** (2 * q))

    if iType[0:] == "BHPF":
        for n in range(N):
            for m in range(M):
                D = np.sqrt((m - m_c) ** 2 + (n - n_c) ** 2)
                oMatrix[n, m] = 1 / (1 + (iD0 / D) ** (2 * q))

    return oMatrix

if __name__ == "__main__":
    H = getFilterSpectrum(freq_image, min(freq_image.shape) / 10, "ILPF")
    analyzeDFT(H, ["scale", "display"], "ILPF")
    filtered_freq_image = freq_image * analyzeDFT(H, ["center"])
    reconstructed_freq_image = computeDFT(filtered_freq_image, inverse=True)
    analyzeDFT(reconstructed_freq_image, ["amplitude", "display"], "Filtrirana slika")

    # Dodatno Naloga 1: Filtriranje z BLPF in BHPF
    D0 = min(freq_image.shape) / 10

    # Butterworthov nizkoprepustni filter (BLPF)
    H_BLPF = getFilterSpectrum(freq_image, D0, "BLPF")
    analyzeDFT(H_BLPF, ["scale", "display"], "BLPF")
    filtered_freq_image_BLPF = freq_image * analyzeDFT(H_BLPF, ["center"])
    reconstructed_freq_image_BLPF = computeDFT(filtered_freq_image_BLPF, inverse=True)
    analyzeDFT(reconstructed_freq_image_BLPF, ["amplitude", "display"], "Filtrirana slika z BLPF")

    # Butterworthov visokoprepustni filter (BHPF)
    H_BHPF = getFilterSpectrum(freq_image, D0, "BHPF")
    analyzeDFT(H_BHPF, ["scale", "display"], "BHPF")
    filtered_freq_image_BHPF = freq_image * analyzeDFT(H_BHPF, ["center"])
    reconstructed_freq_image_BHPF = computeDFT(filtered_freq_image_BHPF, inverse=True)
    analyzeDFT(reconstructed_freq_image_BHPF, ["amplitude", "display"], "Filtrirana slika z BHPF")

    # Primerjava z ILPF in IHPF
    H_ILPF = getFilterSpectrum(freq_image, D0, "ILPF")
    analyzeDFT(H_ILPF, ["scale", "display"], "ILPF")
    filtered_freq_image_ILPF = freq_image * analyzeDFT(H_ILPF, ["center"])
    reconstructed_freq_image_ILPF = computeDFT(filtered_freq_image_ILPF, inverse=True)
    analyzeDFT(reconstructed_freq_image_ILPF, ["amplitude", "display"], "Filtrirana slika z ILPF")

    H_IHPF = getFilterSpectrum(freq_image, D0, "IHPF")
    analyzeDFT(H_IHPF, ["scale", "display"], "IHPF")
    filtered_freq_image_IHPF = freq_image * analyzeDFT(H_IHPF, ["center"])
    reconstructed_freq_image_IHPF = computeDFT(filtered_freq_image_IHPF, inverse=True)
    analyzeDFT(reconstructed_freq_image_IHPF, ["amplitude", "display"], "Filtrirana slika z IHPF")
   
   # Slike filtrirane z BLPF in BHPF so bolj gladke in imajo manj artefaktov v primerjavi s slikami filtriranimi z ILPF in IHPF

# Dodatno Naloga 3:
if __name__ == "__main__":
    # Nalaganje slik
    image1 = loadImage("vaja10/data/pattern-236x330-08bit.raw", [236, 330], np.uint8)
    image2 = loadImage("vaja3/data/pumpkin-200x152-08bit.raw", [200, 152], np.uint8)
    image3 = loadImage("vaja7/data/cameraman-256x256-08bit.raw", [256, 256], np.uint8)

    # Prikaz originalnih slik
    displayImage(image1, "Originalna slika 1")
    displayImage(image2, "Originalna slika 2")
    displayImage(image3, "Originalna slika 3")

    # Izračun DFT za vse slike
    freq_image1 = computeDFT(image1)
    freq_image2 = computeDFT(image2)
    freq_image3 = computeDFT(image3)

    # Prikaz spektralnih komponent
    analyzeDFT(freq_image1, ["amplitude", "center", "log", "scale", "display"], "Amplitudni spekter slike 1")
    analyzeDFT(freq_image2, ["amplitude", "center", "log", "scale", "display"], "Amplitudni spekter slike 2")
    analyzeDFT(freq_image3, ["amplitude", "center", "log", "scale", "display"], "Amplitudni spekter slike 3")

    # Prikaz enosmerne spektralne komponente
    dc_component1 = freq_image1[0, 0]
    dc_component2 = freq_image2[0, 0]
    dc_component3 = freq_image3[0, 0]

    print(f"Enosmerna spektralna komponenta slike 1: {dc_component1}")
    print(f"Enosmerna spektralna komponenta slike 2: {dc_component2}")
    print(f"Enosmerna spektralna komponenta slike 3: {dc_component3}")

    # Izračun povprečne sivinske vrednosti
    avg_gray_value1 = np.mean(image1)
    avg_gray_value2 = np.mean(image2)
    avg_gray_value3 = np.mean(image3)

    print(f"Povprečna sivinska vrednost slike 1: {avg_gray_value1}")
    print(f"Povprečna sivinska vrednost slike 2: {avg_gray_value2}")
    print(f"Povprečna sivinska vrednost slike 3: {avg_gray_value3}")

    # Primerjava enosmerne spektralne komponente in povprečne sivinske vrednosti
    print(f"Razlika za sliko 1: {np.abs(dc_component1 - avg_gray_value1)}")
    print(f"Razlika za sliko 2: {np.abs(dc_component2 - avg_gray_value2)}")
    print(f"Razlika za sliko 3: {np.abs(dc_component3 - avg_gray_value3)}")

    # Razlika med enosmerno spektralno komponento in popvrpecno sivinsko vrednostjo je zelo nizka, kar potrjuje,
    # da enosmerna spektralna komponenta vsebuje informacijo o povprecni sivinski vrednosti slike.