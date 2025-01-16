import numpy as np
import matplotlib.pyplot as plt
import os, sys

parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)

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


# Naloga 2:
def rgb2hsv(iImageRGB):   
    ri = iImageRGB[:, :, 0]
    gi = iImageRGB[:, :, 1]
    bi = iImageRGB[:, :, 2]
    
    vi = iImageRGB.max(axis=2) 
    ci = vi - iImageRGB.min(axis=2)
    li = vi - ci / 2

    hi = np.zeros_like(vi)
    mask = (vi == ri) & (ci != 0)
    hi[mask] = 60 * ((gi[mask] - bi[mask]) / ci[mask])
    mask = (vi == gi) & (ci != 0)
    hi[mask] = 60 * (2 + (bi[mask] - ri[mask]) / ci[mask])
    mask = (vi == bi) & (ci != 0)
    hi[mask] = 60 * (4 + (ri[mask] - gi[mask]) / ci[mask])
    hi[ci == 0] = 0

    si = np.zeros_like(vi)
    si[vi != 0] = ci[vi != 0] / vi[vi != 0]

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

    h_slice[h_slice < 100] = h_slice[h_slice < 100] / 2

    img_hsv_transformed = np.stack((h_slice, s_slice, v_slice), axis=2)

    plt.imshow(img_hsv_transformed)
    plt.show()

    print(np.array_equal(hsv_image, img_hsv_transformed)) 

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

    mask = (0 <= hi_hat) & (hi_hat < 1)
    ri[mask] = ci[mask]
    gi[mask] = xi[mask]

    mask = (1 <= hi_hat) & (hi_hat < 2)
    ri[mask] = xi[mask]
    gi[mask] = ci[mask]

    mask = (2 <= hi_hat) & (hi_hat < 3)
    gi[mask] = ci[mask]
    bi[mask] = xi[mask]

    mask = (3 <= hi_hat) & (hi_hat < 4)
    gi[mask] = xi[mask]
    bi[mask] = ci[mask]

    mask = (4 <= hi_hat) & (hi_hat < 5)
    ri[mask] = xi[mask]
    bi[mask] = ci[mask]

    mask = (5 <= hi_hat) & (hi_hat < 6)
    ri[mask] = ci[mask]
    bi[mask] = xi[mask]

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