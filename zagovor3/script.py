import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import displayImage

# Naloga 1:
def standardize_image(iImage):
    iImage = iImage.astype(float)
    oImage = np.copy(iImage)
    
    # Red
    red = iImage[:, :, 0]
    red_mean = red.mean()
    red_std = red.std()
    red = (red - red_mean) / red_std

    # Green
    green = iImage[:, :, 1]
    green_mean = green.mean()
    green_std = green.std()
    green = (green - green_mean) / green_std

    # Blue
    blue = iImage[:, :, 2]
    blue_mean = blue.mean()
    blue_std = blue.std()
    blue = (blue - blue_mean) / blue_std

    oImage[:, :, 0] = red
    oImage[:, :, 1] = green
    oImage[:, :, 2] = blue


    return oImage

if __name__ == "__main__":
    image = plt.imread("zagovor3/data/planina-uint8.jpeg")
    standardized_image = standardize_image(image)
    displayImage(standardized_image, 'Standardizirana slika')

# Naloga 2:
def rgb2hsl(iRGB):
    red = iRGB[:, :, 0]
    green = iRGB[:, :, 1]
    blue = iRGB[:, :, 2]

    v = iRGB.max(-1)
    c = iRGB.max(-1) - iRGB.min()
    l = v - c/2

    h = np.zeros_like(v)
    h[v == red] = 60 * ((green - blue) / c )[v == red]
    h[v == green] = 60 * (2 + (blue - red) / c )[v == green]
    h[v == blue] = 60 * (4 + (red - green) / c )[v == blue]
    h[c == 0] = 0

    s_l = np.zeros_like(l)
    mask = np.logical_or(l == 0, l == 1)
    s_l[~mask] = (c / (1 - np.abs(2 * v - c - 1)))[~mask]

    oHSL = np.stack((h, s_l, l), axis=-1)
 

    return oHSL

if __name__ == "__main__":
    hsl_image = rgb2hsl(image)
    displayImage(hsl_image, 'HSL slika')

# Naloga 3:
if __name__ == "__main__":
    hsl_image_modified = hsl_image
    h_slice = hsl_image[:, :, 0]
    h_slice = np.where((h_slice >= 100) & (h_slice < 250), h_slice / 5, h_slice)

    hsl_image_modified[:, :, 0] = h_slice
    displayImage(hsl_image_modified, 'Filtrirana HSL slika')

# Naloga 4:
def hsl2rgb(iHSL):
    h = iHSL[:, :, 0]
    s = iHSL[:, :, 1]
    l = iHSL[:, :, 2]

    c = (1 - np.abs(2 * l - 1)) * s
    h_hat = h / 60
    x = c * (1 - np.abs(h_hat % 2 - 1))

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    mask = np.logical_and(0 <= h_hat, h_hat < 1)
    r[mask] = c[mask]
    g[mask] = x[mask]

    mask = np.logical_and(1 <= h_hat, h_hat < 2)
    r[mask] = x[mask]
    g[mask] = c[mask]

    mask = np.logical_and(2 <= h_hat, h_hat < 3)
    g[mask] = c[mask]
    b[mask] = x[mask]

    mask = np.logical_and(3 <= h_hat, h_hat < 4)
    g[mask] = x[mask]
    b[mask] = c[mask]

    mask = np.logical_and(4 <= h_hat, h_hat < 5)
    r[mask] = x[mask]
    b[mask] = c[mask]

    mask = np.logical_and(5 <= h_hat, h_hat < 6)
    r[mask] = c[mask]
    b[mask] = x[mask]

    m = l - c / 2
    oRGB = np.stack((r + m, g + m, b + m), axis=-1)


    return oRGB

if __name__ == "__main__":
    rgb_image = hsl2rgb(hsl_image)
    displayImage(rgb_image, 'RGB slika')
    rgb_image2 = hsl2rgb(hsl_image_modified)
    displayImage(rgb_image2, 'RGB filtrirane slike')