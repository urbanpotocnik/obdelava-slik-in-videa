import matplotlib.pyplot as plt
import numpy as np

# Naloga 1
if __name__ == '__main__':
    im = plt.imread('/home/urban/Faks/Obdelava slik in videa/Vaje/vaja1/data/lena-color.png')

    plt.figure()
    plt.imshow(im)
    plt.show()
    
    plt.imsave('/home/urban/Faks/Obdelava slik in videa/Vaje/vaja1/data/lena-color.png', im)

# Naloga 2
def loadImage(iPath, iSize, iType):
    with open(iPath, 'rb') as fid:
        buffer = fid.read()

    buffer_len = len(np.frombuffer(buffer=buffer, dtype=iType))
    
    if buffer_len != np.prod(iSize):
        raise ValueError('Size of the data does not match the specified size')
    else:
        oImage_shape = (iSize[1], iSize[0])

    oImage = np.ndarray(oImage_shape, dtype = iType, buffer = buffer, order = 'F')
    return oImage

# Naloga 3
def displayImage(iImage, iTitle=''):
    fig = plt.figure()
    plt.title(iTitle)
    plt.imshow(iImage, cmap='gray', vmin=0, vmax=255, aspect='equal')
    return fig

if __name__ == '__main__':
    image__2__gray = loadImage('vaja1/data/lena-gray-410x512-08bit.raw', (410, 512), np.uint8)
    figure = displayImage(image__2__gray, 'Lena')
    plt.show()

# Dodatek
def saveImage(iImage, iPath, iType):
    with open(iPath, 'wb') as fid:      # 'rb'=read mode, 'wb'=write mode
        fid.write(iImage.tobytes())
        fid.close()                     # there is no need for the close method, but it is written in the manual
