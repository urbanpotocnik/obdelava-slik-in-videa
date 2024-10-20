import numpy as np
import matplotlib.pyplot as plt

from OSV_lib import displayImage, loadImage

# Naloga 1
if __name__ == "__main__":
    image = loadImage("/home/urban/Faks/Obdelava slik in videa/Vaje/vaja2/data/valley-1024x683-08bit.raw", (1024, 683), np.uint8)
    displayImage(image, "Originalna slika")

# Naloga 2
def computeHistorgram(iImage):
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

if __name__ == "__main__":
    hist, prob, CDF, levels = computeHistorgram(image)
    displayHistogram(hist, levels, "Histogram")
    displayHistogram(prob, levels, "Normaliziran histogram")
    displayHistogram(CDF, levels, "CDF")

# Naloga 3
def equalizeHistogram(iImage):
    _, _, CDF, _ = computeHistorgram(iImage)    # Izračun CDFja iz podane slike

    nBits = int(np.log2(iImage.max())) + 1
    
    max_intensity = 2 ** nBits + 1

    oImage = np.zeros_like(iImage)              # Ustvari isto sliko kot je iImage, vendar vse piksle nastavi na 0

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):        # Novo intenziteto izračunamo tako, da uporabimo vrednost CDF za staro intenziteto in jo pomnožimo z največjo možno intenziteto max_intensity
            old_intensity = iImage[y, x]
            new_intensity = np.floor(CDF[old_intensity] * max_intensity)
            oImage[y, x] = new_intensity
    
    return oImage

if __name__ == "__main__":
    image_equalized = equalizeHistogram(image)
    displayImage(image_equalized, "Slika z izravnanim histogramom")
    hist, prob, CDF, levels = computeHistorgram(image_equalized)
    displayHistogram(hist, levels, "Histogram izravnane slike")
    displayHistogram(CDF, levels, "CDF izravnane slike")

# Dodatno: Naloga 3
# Entropija slike je mera za količino informacije, ki jo vsebuje slika
def computeEntropy(iImage):
    hist, _, _, _  = computeHistorgram(iImage)
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

if __name__ == "__main__":
    normalPictureEntropy = computeEntropy(image)
    equalizedPictureEntropy = computeEntropy(image_equalized)

    print(f"Entropija navadne slike = {normalPictureEntropy}")
    print(f"Entropija izravnane slike = {equalizedPictureEntropy}")
    
'''
Večja bo entropija izravnane slike, ker izravnava histograma povečuje razpršenost vrednosti pikslov, 
kar vodi do bolj enakomerne porazdelitve in posledično višje entropije.
'''
    
# Dodatno: Naloga 4
def addNoise(iImage, iStd):
    height, width = iImage.shape
    oNoise = np.random.randn(height, width) * iStd
    iImage = iImage.astype(np.uint8)
    oImage = oNoise + iImage
    
    return oImage, oNoise

if __name__ == "__main__":
    displayImage(image, "Originalna slika brez noisa")

    for i in [2, 5, 10, 25]:
        noisyImage, _ = addNoise(image, i)
        displayImage(noisyImage, f"Originalna slika z noisom, standardni odklon = {i}")
        
'''
Slika šuma lahko vsebuje negativne in pozitivne vrednosti (ker Gaussov šum vsebuje vrednosti okoli ničle). 
To lahko povzroči težave pri prikazovanju, saj slike običajno pričakujejo vrednosti med 0 in 255 za sivinsko lestvico.
    
Pri računanju histograma šuma moraš upoštevati, da histogram morda vključuje tudi vrednosti, 
ki presegajo običajne meje slike (pod 0 ali nad 255), tako da moramo prikazati histogram, 
ki prikazuje dejansko porazdelitev šuma brez omejitev na vrednosti 0–255, saj bi omejevanje popačilo rezultate.
'''