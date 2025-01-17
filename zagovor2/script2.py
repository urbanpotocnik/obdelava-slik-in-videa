import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import displayImage, spatialFiltering

# Naloga 1:
def color2grayscale(iImage):
    normalized_image = (iImage.astype(float) / iImage.max()) * 255
    grayscale_image = np.mean(normalized_image, axis=2)    
    oImage = np.floor(grayscale_image).astype(np.uint8)
    
    return oImage


if __name__ == "__main__":
    image = plt.imread("zagovor2/data/paris_map-807-421.png")
    grayscale_image = color2grayscale(image)
    plt.imshow(grayscale_image, cmap='gray')
    plt.show()


# Naloga 2:
def transformImage(iImage, angle, center):
    """
    Rotacija slike okoli določene točke z linearno interpolacijo.
    """
    Y, X = iImage.shape
    oImage = np.zeros_like(iImage)
    
    Ttrans = np.array([[1, 0, -center[0]],
                       [0, 1, -center[1]],
                       [0, 0, 1]])
    
    angle_rad = np.radians(angle)
    Trot = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                     [np.sin(angle_rad), np.cos(angle_rad), 0],
                     [0, 0, 1]])
    
    Ttrans_inv = np.array([[1, 0, center[0]],
                           [0, 1, center[1]],
                           [0, 0, 1]])
    
    T = Ttrans_inv @ Trot @ Ttrans
    
    T_inv = np.linalg.inv(T)
    
    for y in range(Y):
        for x in range(X):
            pt = np.array([x, y, 1])
            
            pt_transformed = T_inv @ pt
            x_t, y_t = pt_transformed[:2]
            
            if 0 <= x_t < X-1 and 0 <= y_t < Y-1:
                x0, y0 = int(np.floor(x_t)), int(np.floor(y_t))
                x1, y1 = x0 + 1, y0 + 1
                
                a = x_t - x0
                b = y_t - y0
                
                oImage[y, x] = (1 - a) * (1 - b) * iImage[y0, x0] + \
                               a * (1 - b) * iImage[y0, x1] + \
                               (1 - a) * b * iImage[y1, x0] + \
                               a * b * iImage[y1, x1]
    
    return oImage


if __name__ == "__main__":
    image = plt.imread("zagovor2/data/paris_map-807-421.png")
    grayscale_image = color2grayscale(image)
    
    A = np.array([354, 156])
    B = np.array([400, 171])
    
    a = B - A
    b = np.array([1, 0])
    
    skalarni_produkt_a_b = np.dot(a, b)
    
    dolzina_a = np.linalg.norm(a)
    dolzina_b = np.linalg.norm(b)
    
    cos_phi_rad = skalarni_produkt_a_b / (dolzina_a * dolzina_b)
    phi = np.arccos(cos_phi_rad)
    phi_degrees = np.degrees(phi)
    
    print(f"Kot φ med vektorjema a in b je {phi_degrees:.2f} stopinj.")
    
    rotated_image = transformImage(grayscale_image, -phi_degrees, A)
    
    plt.imshow(rotated_image, cmap='gray')
    plt.show()


# Naloga 3:
if __name__ == "__main__":
    """
    Izračun robov z uporabo Sobelovega operatorja.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Y, X = rotated_image.shape
    
    gradient_x = np.zeros_like(rotated_image, dtype=float)
    gradient_y = np.zeros_like(rotated_image, dtype=float)
    
    for y in range(1, Y-1):
        for x in range(1, X-1):
            region = rotated_image[y-1:y+2, x-1:x+2]
            gradient_x[y, x] = np.sum(region * sobel_x)
            gradient_y[y, x] = np.sum(region * sobel_y)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    
    edges = gradient_magnitude
    
    threshold = 100
    edges[edges < threshold] = 0
    edges[edges >= threshold] = 255
    
    plt.imshow(edges, cmap='gray')
    plt.show()


# Naloga 4:
def getSquareCenterPoint(iImage, iLength):
    """
    Določi koordinate središča kvadrata z dano dolžino stranice z uporabo Houghove preslikave.
    """
    Y, X = iImage.shape
    oAcc = np.zeros((Y, X))

    rangeF = np.arange(0, 360, 1)  
    rangeFrad = np.deg2rad(rangeF)  
    idxF = np.arange(len(rangeF))

    for y in range(Y):
        for x in range(X):
            if iImage[y, x]:
                for f_idx in idxF:
                    fi = rangeFrad[f_idx]
                    x0 = int(x - iLength / 2 * np.cos(fi))
                    y0 = int(y - iLength / 2 * np.sin(fi))

                    if 0 <= x0 < X and 0 <= y0 < Y:
                        oAcc[y0, x0] += 1

    max_value = 0
    oCenter = [0, 0]
    for y in range(Y):
        for x in range(X):
            if oAcc[y, x] > max_value:
                max_value = oAcc[y, x]
                oCenter = [x, y]  

    return oCenter, oAcc


if __name__ == "__main__":
    iLength = 53 
    oCenter, oAcc = getSquareCenterPoint(edges, iLength)
    
    plt.imshow(rotated_image, cmap='gray')
    plt.scatter(oCenter[0], oCenter[1], color='red', s=100)
    plt.gca().add_patch(plt.Rectangle((oCenter[0] - iLength // 2, oCenter[1] - iLength // 2), iLength, iLength, edgecolor='red', facecolor='none'))
    plt.show()
