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

    # Normaliziran histogram -> prikaze nam porazdelitev vrednosti
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

'''
if __name__ == "__main__":
    image_equalized = equalizeHistogram(image)
    displayImage(image_equalized, "Slika z izravnanim histogramom")
    hist, prob, CDF, levels = computeHistorgram(image_equalized)
    displayHistogram(hist, levels, "Histogram izravnane slike")
    displayHistogram(CDF, levels, "CDF izravnane slike")

if __name__ == "__main__":
    hist, prob, CDF, levels = computeHistorgram(image)
    displayHistogram(hist, levels, "Histogram")
    displayHistogram(prob, levels, "Normaliziran histogram")
    displayHistogram(CDF, levels, "CDF")
'''

# Naloga 3
def equalizeHistogram(iImage):
    _, _, CDF, _ = computeHistorgram(iImage)    # Izracun CDFja iz podane slike

    nBits = int(np.log2(iImage.max())) + 1
    
    max_intensity = 2 ** nBits + 1

    oImage = np.zeros_like(iImage)              # Ustvari isto sliko kot je iImage, vendar vse piksle nastavi na 0

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):        # Novo intenziteto izracunamo tako, da uporabimo vrednost CDF za staro intenziteto in jo pomnozimo z najvecjo možno intenziteto max_intensity
            old_intensity = iImage[y, x]
            new_intensity = np.floor(CDF[old_intensity] * max_intensity)
            oImage[y, x] = new_intensity
    
    return oImage
'''
if __name__ == "__main__":
    image_equalized = equalizeHistogram(image)
    displayImage(image_equalized, "Slika z izravnanim histogramom")
    hist, prob, CDF, levels = computeHistorgram(image_equalized)
    displayHistogram(hist, levels, "Histogram izravnane slike")
    displayHistogram(CDF, levels, "CDF izravnane slike")
'''


# Dodatno: Naloga 3
# Entropija slike je mera za kolicino informacije, ki jo vsebuje slika
def computeEntropy(iImage):
    hist, _, _, _  = computeHistorgram(iImage)
    height, width = iImage.shape                      # Izracunanje visine in sirine slike
    nPixels = height * width                         
    probabilities = np.zeros_like(hist, dtype=float)  # Inicializacija arraya
    
    probabilities = hist / nPixels

    entropy = 0

    for p in probabilities: 
       if p > 0:                        # Ignoriramo niclo
            entropy += p * np.log2(p)
    
    oEntropy = -entropy

    return oEntropy
'''
 if __name__ == "__main__":
    normalPictureEntropy = computeEntropy(image)
    equalizedPictureEntropy = computeEntropy(image_equalized)

    print(f"Entropija navadne slike = {normalPictureEntropy}")
    print(f"Entropija izravnane slike = {equalizedPictureEntropy}")
    
    Vecja bo entropija izravnane slike, ker izravnava histograma povečuje razprsenost vrednosti pikslov, 
    kar vodi do bolj enakomerne porazdelitve in posledicno visje entropije.
    

'''      

# Dodatno: Naloga 4
'''
def addNoise(iImage, iStd):
    iImage = iImage.astype(np.uint8)        # Pretvorba slike v uint8
    oImage = np.zeros_like(iImage)

    height, width = iImage.shape
    iStd = np.random.randn(height, width)

    randomValues = np.random.randn(height, width)   # Generiramo array z random podatki 
    normalizedValues = (randomValues - randomValues.min()) / (randomValues.max() - randomValues.min()) # Pretvorimo vrednosti na [0, 1] z normalizacijo
    oNoise = normalizedValues * iStd                # Razsirimo na [0, iStd]

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            iImage[y, x] = iImage[y, x] + oNoise[y, x]

    return oImage , oNoise
'''


def addNoise(iImage, iStd):
    # Create a noise matrix using random normal distribution
    noise = np.random.randn(*iImage.shape) * iStd  # Scale the noise by iStd
    
    # Add the noise to the original image
    oImage = iImage.astype(np.float32) + noise  # Convert to float to prevent overflow
    
    # Clip values to stay within valid range [0, 255] for uint8
    oImage = np.clip(oImage, 0, 255).astype(np.uint8)
    
    return oImage, noise

if __name__ == "__main__":
    noisyImage = addNoise(image, 155)
    displayImage(noisyImage, "Originalna slika z noisom")
