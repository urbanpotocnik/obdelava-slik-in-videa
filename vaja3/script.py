import numpy as np 
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)

from OSV_lib import loadImage, displayImage, computeHistogram, displayHistogram 

# Naloga 1
if __name__ == "__main__":
    Image = loadImage("vaja3/data/pumpkin-200x152-08bit.raw", (200, 152), np.uint8)
    displayImage(Image, "Originalna slika")
    plt.show()

# Naloga 2 -> interpolacija slike
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


if __name__ == "__main__":
    intSize = [Image.shape[1] * 2, Image.shape[0] * 2]
    interpolated_0_order = interpolateImage(Image, intSize, 0)
    displayImage(interpolated_0_order, "Interpolirana slika 0 reda")

    interpolated_1_order = interpolateImage(Image, intSize, 1)
    displayImage(interpolated_1_order, "Interpolirana slika 1 reda")

# Dodatno: Naloga 1 -> interpolacija izbranega obmocja
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

if __name__ == "__main__":
    start_coords = (75, 30)
    region_size = (65, 50)
    min_val, max_val, avg_val = analyzeInterpolationRegion(Image, start_coords, region_size)


# Dodatno: Naloga 2 -> interpolacija 0 reda
"""
Prednosti interpolacije ničtega reda:

-Enostavnost: Zelo enostavna za implementacijo.
-Hitrost: Hitro izvajanje, primerno za velike slike ali realnočasne aplikacije.
-Brez izkrivljanja: Ne uvaja dodatnega izkrivljanja slik.

Slabosti interpolacije ničtega reda:

-Kvaliteta slike: Slaba kakovost, s pikslastimi robovi.
-Izguba podrobnosti: Pomembne podrobnosti se lahko izgubijo.
-Zameglitev: Ustvarja zamegljene robove in umetne učinke.
-Neprimerna za natančne aplikacije: Slaba natančnost v znanstvenih raziskavah.
"""

if __name__ == "__main__":
    intSize = [600, 300]
    interpolated_0_order = interpolateImage(Image, intSize, 0)
    displayImage(interpolated_0_order, "Interpolirana slika 0 reda")

    hist, prob, CDF, levels = computeHistogram(interpolated_0_order)
    displayHistogram(hist, levels, "Histogram interpolirane slike 0. reda")
    plt.show()

    # Izračunaj minimalne, maksimalne in povprečne sivinske vrednosti
    min_val = np.min(interpolated_0_order)
    max_val = np.max(interpolated_0_order)
    avg_val = np.mean(interpolated_0_order)

    print(f"Minimalna sivinska vrednost: {min_val}")
    print(f"Maksimalna sivinska vrednost: {max_val}")
    print(f"Povprečna sivinska vrednost: {avg_val}")


# Dodatno: Naloga 3 -> interpolacija 1 reda
"""
Prednosti interpolacije prvega reda:

Boljša kakovost slike: Zmanjšuje pikslaste robove in zameglitve.
Ohranjanje podrobnosti: Bolj natančno upodablja prehode in podrobnosti.
Učinkovitost: Razmeroma hitro izvajanje, primerljivo z ničto redno.
Slabosti interpolacije prvega reda:

Računalska kompleksnost: Potrebuje več izračunov, kar poveča čas izvajanja.
Možnost izkrivljanja: Lahko pride do manjših izkrivljanj ali artefaktov.
Neenakomernost: Težave pri prehodih med intenzitetami, kar vodi do neenakomerne distribucije.
"""

if __name__ == "__main__":
    intSize = [600, 300]
    interpolated_1_order = interpolateImage(Image, intSize, 1)
    displayImage(interpolated_1_order, "Interpolirana slika 1 reda")

    hist, prob, CDF, levels = computeHistogram(interpolated_1_order)
    displayHistogram(hist, levels, "Histogram interpolirane slike 1. reda")
    plt.show()

    min_val = np.min(interpolated_1_order)
    max_val = np.max(interpolated_1_order)
    avg_val = np.mean(interpolated_1_order)

    print(f"Minimalna sivinska vrednost: {min_val}")
    print(f"Maksimalna sivinska vrednost: {max_val}")
    print(f"Povprečna sivinska vrednost: {avg_val}")


# Dodatno: Naloga 4
"""
Interpolacije višjih redov, kot je interpolacija tretjega reda, prinašajo naslednje prednosti:

-Boljša gladkost: Gladki in naravni prehodi med pikslom zmanjšujejo ostrine in pikslaste robove.
-Ohranjanje podrobnosti: Boljše ohranjanje detajlov v kompleksnih strukturah in teksturah.
-Manj izkrivljanj: Boljša obravnava nelinearnih prehodov in zmanjšanje izkrivljanj.
-Fleksibilnost: Prilagoditev metod za specifične potrebe aplikacij.
-Uporaba v natančnih aplikacijah: Koristne v računalniški grafiki, medicinskem slikanju in znanstvenih simulacijah.

Vendar višje redne interpolacije zahtevajo več virov in so kompleksnejše za implementacijo.
"""

# Dodatno: Naloga 5
"""
Ta rešitev omogoča prikaz interpolirane slike v pravilnem fizičnem sorazmerju glede na izvirno območje. 
Ko uporabljamo extent, bo prikaz interpolirane slike dimenzijsko skladen z izvirno sliko, 
kar izboljša vizualizacijo natančnosti interpolacije.
"""
if __name__ == "__main__":

    Image = loadImage("vaja3/data/pumpkin-200x152-08bit.raw", (200, 152), np.uint8)
    
    # Izvedba interpolacije 0. in 1. reda
    intSize_0_order = (600, 300)
    interpolated_0_order = interpolateImage(Image, intSize_0_order, 0)
    intSize_1_order = (600, 300)
    interpolated_1_order = interpolateImage(Image, intSize_1_order, 1)
    
    # Definiranje gridov
    gridX_0_order = np.linspace(0, Image.shape[1] - 1, intSize_0_order[0])
    gridY_0_order = np.linspace(0, Image.shape[0] - 1, intSize_0_order[1])
    gridX_1_order = np.linspace(0, Image.shape[1] - 1, intSize_1_order[0])
    gridY_1_order = np.linspace(0, Image.shape[0] - 1, intSize_1_order[1])
    
    # Prikaz interpolirane slike z ustreznim dimenzionalnim sorazmerjem
    displayImage(interpolated_0_order, "Interpolirana slika 0. reda", gridX_0_order, gridY_0_order)
    displayImage(interpolated_1_order, "Interpolirana slika 1. reda", gridX_1_order, gridY_1_order)


def decimateImage(iImage, iKernel, iLevel):
    """
    Za vsak nivo piramidne decimacije filtriramo sliko s podanim jedrom digitalnega filtra C

    Jedra (kernel) delujejo kot nizkoprepustni filtri, ki zgladijo sliko in zmanjšajo visoke frekvence,
    kar preprečuje aliasing pri nadaljnjem zmanjševanju ločljivosti.

    Po filtriranju vzorčimo sliko z nižjo frekvenco tako, da ohranimo samo vsak drugi piksel v obeh smereh.
    Ta postopek ponovimo za vsako stopnjo (torej dvakrat, če je stopnja 2).

    Nivo piramidne decimacije določa, kolikokrat se postopek filtriranja in vzorčenja ponovi.
    Pri nivoju 2 bomo sliko dvakrat filtrirali in decimirali.
    """

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

if __name__ == "__main__":
    # Podana jedra za M = 1 in M = 2
    kernel_M1 = np.array([
        [1/16, 1/8, 1/16],
        [1/8,  1/4, 1/8],
        [1/16, 1/8, 1/16]
    ])

    kernel_M2 = np.array([
        [1/400, 1/80, 1/50, 1/80, 1/400],
        [1/80,  1/16, 1/10, 1/16, 1/80],
        [1/50,  1/10, 4/25, 1/10, 1/50],
        [1/80,  1/16, 1/10, 1/16, 1/80],
        [1/400, 1/80, 1/50, 1/80, 1/400]
    ])

    Image = loadImage("vaja3/data/pumpkin-200x152-08bit.raw", (200, 152), np.uint8)
    iLevel = 2

    # Decimacija z jedrom M = 1
    decimated_image_M1 = decimateImage(Image, kernel_M1, iLevel)
    displayImage(decimated_image_M1, "Decimirana slika M=1, nivo=2")

    # Decimacija z jedrom M = 2
    decimated_image_M2 = decimateImage(Image, kernel_M2, iLevel)
    displayImage(decimated_image_M2, "Decimirana slika M=2, nivo=2")