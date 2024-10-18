import matplotlib.pyplot as plt
import numpy as np

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

def displayImage(iImage, iTitle=''):
    fig = plt.figure()
    plt.title(iTitle)
    plt.imshow(iImage, cmap='gray', vmin=0, vmax=255, aspect='equal')
    plt.show()
    return fig

# Dodatek
def saveImage(iImage, iPath, iType):
    with open(iPath, 'wb') as fid:      # 'rb'=read mode, 'wb'=write mode
        fid.write(iImage.tobytes())
        fid.close()                     # there is no need for the close method, but it is written in the manual

# TO DO tukaj dodaj vse metode, use tudi malo pokomentiraj
# dodaj se importe 