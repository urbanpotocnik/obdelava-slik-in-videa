import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import displayImage


# Naloga 1:
def standardize_image(iImage):
    iImage = iImage.astype(np.float32)

    red = iImage[:,:,0]
    green = iImage[:,:,1]
    blue = iImage[:,:,2]

    red_slice = (red - np.mean(red)) / np.std(red)
    green_slice = (green - np.mean(green)) / np.std(green)
    blue_slice = (blue - np.mean(blue)) / np.std(blue)

    oImage = np.stack((red_slice, green_slice, blue_slice), axis=2)
    oImage = ((oImage - oImage.min()) / (oImage.max() - oImage.min()) * 255).astype(np.uint8)

    return oImage


if __name__ == "__main__":
    iImage = plt.imread("zagovor3/data/planina-uint8.jpeg")
    img_std = standardize_image(iImage)
    displayImage(img_std, 'Standardizirana slika')


# Naloga 2:
def rgb2hsl(iRGB):
    ri = iRGB[:, :, 0] / 255.0
    gi = iRGB[:, :, 1] / 255.0
    bi = iRGB[:, :, 2] / 255.0

    vi = np.max(iRGB, axis=2) / 255.0
    ci = vi - np.min(iRGB, axis=2) / 255.0
    li = (vi + np.min(iRGB, axis=2) / 255.0) / 2

    hi = np.zeros_like(vi)
    mask = (ci != 0)
    hi[mask & (vi == ri)] = 60 * (gi[mask & (vi == ri)] - bi[mask & (vi == ri)]) / ci[mask & (vi == ri)]
    hi[mask & (vi == gi)] = 60 * (2 + (bi[mask & (vi == gi)] - ri[mask & (vi == gi)]) / ci[mask & (vi == gi)])
    hi[mask & (vi == bi)] = 60 * (4 + (ri[mask & (vi == bi)] - gi[mask & (vi == bi)]) / ci[mask & (vi == bi)])
    hi[hi < 0] += 360
    hi[ci == 0] = 0

    si = np.zeros_like(vi)
    si[mask] = ci[mask] / (1 - np.abs(2 * li[mask] - 1))

    hsl_image = np.stack((hi, si, li), axis=2)

    return hsl_image
    

if __name__ == "__main__":
    img_hsl = rgb2hsl(img_std)
    displayImage(img_hsl, 'HSL slika')


# Naloga 3:
if __name__ == "__main__":
    h_slice = img_hsl[:, :, 0].copy()  
    mask = (h_slice >= 100) & (h_slice < 250)
    h_slice[mask] = h_slice[mask] / 5

    s_slice = img_hsl[:, :, 1]
    l_slice = img_hsl[:, :, 2]

    img_hsl_transformed = np.stack((h_slice, s_slice, l_slice), axis=2)
    displayImage(img_hsl_transformed, 'Transformirana HSL slika')


# Naloga 4:
def hsl2rgb(iHSL):
    h, s, l = iHSL[:, :, 0], iHSL[:, :, 1], iHSL[:, :, 2]
    c = (1 - np.abs(2 * l - 1)) * s
    h_prime = h / 60
    x = c * (1 - np.abs(h_prime % 2 - 1))
    
    r1 = np.zeros_like(h)
    g1 = np.zeros_like(h)
    b1 = np.zeros_like(h)
    
    mask = (0 <= h_prime) & (h_prime < 1)
    r1[mask], g1[mask], b1[mask] = c[mask], x[mask], 0
    
    mask = (1 <= h_prime) & (h_prime < 2)
    r1[mask], g1[mask], b1[mask] = x[mask], c[mask], 0
    
    mask = (2 <= h_prime) & (h_prime < 3)
    r1[mask], g1[mask], b1[mask] = 0, c[mask], x[mask]
    
    mask = (3 <= h_prime) & (h_prime < 4)
    r1[mask], g1[mask], b1[mask] = 0, x[mask], c[mask]
    
    mask = (4 <= h_prime) & (h_prime < 5)
    r1[mask], g1[mask], b1[mask] = x[mask], 0, c[mask]
    
    mask = (5 <= h_prime) & (h_prime < 6)
    r1[mask], g1[mask], b1[mask] = c[mask], 0, x[mask]
    
    m = l - c / 2
    r, g, b = r1 + m, g1 + m, b1 + m
    
    oRGB = np.stack((r, g, b), axis=2)
    return oRGB


if __name__ == "__main__":
    img_rgb = hsl2rgb(img_hsl)
    displayImage(img_rgb, 'RGB slika')

    img_rgb_transformed = hsl2rgb(img_hsl_transformed)
    displayImage(img_rgb_transformed, 'Transformirana RGB slika')