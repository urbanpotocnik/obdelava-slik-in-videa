import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import spatialFiltering, displayImage

# Naloga 1:
def get_blue_region(iImage , iThreshold):
    image = iImage

    blue_slice = np.zeros_like(image[:,:,2])
    blue_condition = image[:,:,2] > iThreshold
    white_condition = (image[:,:,0] > iThreshold) & (image[:,:,1] > iThreshold) & (image[:,:,2] > iThreshold)

    blue_slice[blue_condition & ~white_condition] = 255
    oImage = blue_slice

    return oImage


if __name__ == "__main__":
   image = plt.imread("zagovor5/data/bled-lake-decimated-uint8.jpeg")
   plt.imshow(image)
   plt.show()

   blue_image = get_blue_region(image, 235)
   plt.imshow(blue_image, cmap='gray')
   plt.show()


# Naloga 2:
if __name__ == "__main__":
    SE3 = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])

    SE7 = np.array([[0,0,0,1,0,0,0],
                    [0,0,1,1,1,0,0],
                    [0,1,1,1,1,1,0],
                    [1,1,1,1,1,1,1],
                    [0,1,1,1,1,1,0],
                    [0,0,1,1,1,0,0],
                    [0,0,0,1,0,0,0],])

    MImage = spatialFiltering("morphological", blue_image, iFilter=SE7, iMorphOp="erosion")
    lake_mask = spatialFiltering("morphological", MImage, iFilter=SE7, iMorphOp="dialation")

    plt.imshow(lake_mask, cmap='gray')
    plt.show()

    # Zaznava robov = dilation - mask
    dilated = spatialFiltering("morphological", lake_mask, iFilter=SE3, iMorphOp="dialation")

    lake_edge_mask = dilated - lake_mask
    plt.imshow(lake_edge_mask, cmap='gray')
    plt.show()


# Naloga 3:
def find_edge_coordinates(iImage):
    edge_array = np.empty((0, 2), int)

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            if iImage[y,x] == 255:
                edge_array = np.append(edge_array, [[x,y]], axis=0)

    return edge_array


if __name__ == "__main__":
    edge_array = find_edge_coordinates(lake_edge_mask)
    print(edge_array)


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
