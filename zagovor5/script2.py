import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import spatialFiltering, displayImage

# Naloga 1:
def get_blue_region(iImage, iThreshold):
    oImage = np.zeros_like(iImage[:,:,2])
    
    blue_part = np.zeros_like(iImage[:,:,2])
    blue = iImage[:,:,2] > iThreshold
    white = (iImage[:,:,0] > iThreshold) & (iImage[:,:,1] > iThreshold) & (iImage[:,:,2] > iThreshold)

    blue_part[blue & ~white] = 255
    oImage = blue_part

    return oImage


if __name__ == "__main__":    
    iImage = plt.imread("zagovor5/data/bled-lake-decimated-uint8.jpeg")
    iThreshold = 235
    blue_region = get_blue_region(iImage, iThreshold)
    plt.imshow(blue_region, cmap='gray')
    plt.show()

# Naloga 2:
if __name__ == "__main__":
    # erozija nato dilacija
    MKernel = np.array([
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]
        ])
        
    erosion = spatialFiltering("morphological", blue_region, iFilter=MKernel, iMorphOp="erosion")
    dilation = spatialFiltering("morphological", erosion, iFilter=MKernel, iMorphOp="dialation")
    lake_mask = dilation

    displayImage(lake_mask, "Lake mask")

    # zaznava robov = dilacija - mask
    dilated = spatialFiltering("morphological", lake_mask, iFilter=MKernel, iMorphOp="dialation")
    lake_edge_mask = dilated - lake_mask
    plt.imshow(lake_edge_mask, cmap='gray')
    plt.show()


# Naloga 3:
def find_edge_coordinates(iImage):
    oEdges = []
    y, x = iImage.shape

    for y in range(y):
        for x in range(x):
            if iImage[y, x] == 255:
                oEdges.append((y, x))

    return oEdges


if __name__ == "__main__":
    edge_coordinates = find_edge_coordinates(lake_edge_mask)
    print(len(edge_coordinates))


# Naloga 4:
def compute_distances(iImage , iMask = None):
    oImage = np.zeros_like(iImage)

    Y, X = iImage.shape
    edges = find_edge_coordinates(lake_edge_mask)

    for x in range(X):
        for y in range(Y):
            if iMask is None or iMask[y,x]:
                # Racunanje razdalje do najblizje tocke roba
                distances = ( (edges[:,0] - x)**2 + (edges[:,1] - y)**2 )**0.5
                oImage[y,x] = np.min(distances)

    return oImage


if __name__ == "__main__":
    dist = compute_distances(lake_edge_mask, lake_mask)

    loc_max = np.unravel_index(dist.argmax(), dist.shape)
    print(f"lokacija maximuma: {loc_max[1], loc_max[0]}, oddaljenost od obale: {dist[loc_max]}")
    dist = 255*dist/np.max(dist)
    image_distances = dist + lake_edge_mask
    
    plt.imshow(image_distances, cmap='gray')
    plt.plot(loc_max[1], loc_max[0], "rx")  
    plt.show()