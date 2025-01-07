import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
#from OSV_lib import 

# Naloga 1:
if __name__ == "__main__":
    image = plt.imread("zagovor4/data/travnik-uint8.jpeg")
    plt.imshow(image)
    plt.show()

def normalize_image(iImage):
    normalized_image = iImage.astype(float) / iImage.max()
    return normalized_image

if __name__ == "__main__":
    normalized_image = normalize_image(image)
    plt.imshow(normalized_image)
    plt.show() 
    print(normalized_image.max(), image.max())
    print(normalized_image.min(), image.min())
    # Preveri po korakih in natancno preberi navodila

# Naloga 2:
def rgb2hsv(iImageRGB):   
    ri = iImageRGB[:, :, 0]
    gi = iImageRGB[:, :, 1]
    bi = iImageRGB[:, :, 2]
    
    vi = iImageRGB.max(axis=2) # Gledamo max po 3 dimenzji (barve)
    ci = vi - iImageRGB.min(axis=2)
    li = vi - ci / 2

    hi = np.zeros_like(vi)
    hi[vi == ri] = 60 * (gi[vi == ri] - bi[vi == ri]) / ci[vi == ri]
    hi[vi == gi] = 60 * (2 + (bi[vi == gi] - ri[vi == gi]) / ci[vi == gi])
    hi[vi == bi] = 60 * (4 + (ri[vi == bi] - gi[vi == bi]) / ci[vi == bi])
    hi[ci == 0] = 0

    si = np.zeros_like(vi)
    si = ci / vi
    si[vi == 0] = 0
    
    hsv_image = np.stack((hi, si, vi), axis=2)

    return hsv_image


if __name__ == "__main__":
    hsv_image = rgb2hsv(normalize_image(image))
    plt.imshow(hsv_image)
    plt.show()


# Naloga 3:
if __name__ == "__main__":
    h_slice = hsv_image[:, :, 0].copy()
    s_slice = hsv_image[:, :, 1]
    v_slice = hsv_image[:, :, 2]

    #plt.imshow(h_slice)
    #plt.show()
    h_slice[h_slice < 100] = h_slice[h_slice < 100] / 2
    #plt.imshow(h_slice)
    #plt.show()

    img_hsv_transformed = np.stack((h_slice, s_slice, v_slice), axis=2)

    plt.imshow(img_hsv_transformed)
    plt.show()

    #plt.imshow(img_hsv_transformed - hsv_image)

    print(np.array_equal(hsv_image, img_hsv_transformed)) # Ce je true smo nekaj naredili narobe (slike sta iste)

# Naloga 4:
def hsv2rgb(hsvImage):
    hi = hsvImage[:, :, 0]
    si = hsvImage[:, :, 1]
    vi = hsvImage[:, :, 2]

    ci = si * vi
    hi_hat = hi / 60
    xi = ci * (1 - np.abs(hi_hat % 2 - 1))

    ri = np.zeros_like(hi)
    gi = np.zeros_like(hi)
    bi = np.zeros_like(hi)

    ri[(0 <= hi_hat) & (hi_hat < 1)] = ci[(0 <= hi_hat) & (hi_hat < 1)] 
    gi[(0 <= hi_hat) & (hi_hat < 1)] = xi[(0 <= hi_hat) & (hi_hat < 1)] 

    ri[(1 <= hi_hat) & (hi_hat < 2)] = xi[(1 <= hi_hat) & (hi_hat < 2)] 
    gi[(1 <= hi_hat) & (hi_hat < 2)] = ci[(1 <= hi_hat) & (hi_hat < 2)] 

    gi[(2 <= hi_hat) & (hi_hat < 3)] = ci[(2 <= hi_hat) & (hi_hat < 3)] 
    bi[(2 <= hi_hat) & (hi_hat < 3)] = xi[(2 <= hi_hat) & (hi_hat < 3)] 

    gi[(3 <= hi_hat) & (hi_hat < 4)] = xi[(3 <= hi_hat) & (hi_hat < 4)] 
    bi[(3 <= hi_hat) & (hi_hat < 4)] = ci[(3 <= hi_hat) & (hi_hat < 4)]

    ri[(4 <= hi_hat) & (hi_hat < 5)] = xi[(4 <= hi_hat) & (hi_hat < 5)] 
    bi[(4 <= hi_hat) & (hi_hat < 5)] = ci[(4 <= hi_hat) & (hi_hat < 5)]

    ri[(5 <= hi_hat) & (hi_hat < 6)] = ci[(5 <= hi_hat) & (hi_hat < 6)] 
    bi[(5 <= hi_hat) & (hi_hat < 6)] = xi[(5 <= hi_hat) & (hi_hat < 6)]

    m = vi - ci

    rgbImage = np.stack((ri + m, gi + m, bi + m), axis=2)

    return rgbImage


if __name__ == "__main__":
    rgb_image = hsv2rgb(hsv_image)
    plt.imshow(rgb_image)
    plt.show()

    rgb_image2 = hsv2rgb(img_hsv_transformed)
    plt.imshow(rgb_image2)
    plt.show()


