import numpy as np
import matplotlib.pyplot as plt
import os, sys
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage

# Naloga 1:
if __name__ == "__main__":
    imSize = [256,512]
    pxDim = [2,1]

    gX = np.arange(imSize[0])*pxDim[0]
    gY = np.arange(imSize[1])*pxDim[1]
    I = loadImage("vaja6/data/lena-256x512-08bit.raw", imSize, np.uint8)
    displayImage(I, "Originalna slika", gX,gY)

# Naloga 2:
def getParameters(iType, scale = None, trans = None, rot = None, shear = None, orig_pts = None, mapped_pts = None):
    # za afino preslikavo: scale je vektor skaliranja oz. povecave, trans je vektor translacije oz. premika, rot = φ je kot rotacije
    # oz. vrtenja v kotnih stopinjah in shear je vektor striga oz. zatega 

    # za radialno preslikavo: orig_pts je matrika kontrolnih tock in mapped_pts je matrika preslikanih kontrolnih tock

    oP = {}

    if iType == "affine":
        if scale is None:
            scale = [1,1]
        if trans is None:
            trans = [0,0]
        if rot is None:
            rot = 0
        if shear is None:
            shear = [0,0]

        Tk = np.array([
            [scale[0],0,0],
            [0,scale[1],0],
            [0,0,1]
        ])

        Tt = np.array([
            [1,0,trans[0]],
            [0,1,trans[1]],
            [0,0,1]
        ])

        phi = rot*np.pi / 180

        Tf = np.array([
            [np.cos(phi),-np.sin(phi),0],
            [np.sin(phi),np.cos(phi),0],
            [0,0,1]
        ])

        Tg = np.array([
            [1,shear[0],0],
            [shear[1],1,0],
            [0,0,1]
        ])        

        oP = Tg @ Tf @ Tt @ Tk

    elif iType == "radial":
        assert orig_pts is not None, "Manjkajo orig_pts"
        assert mapped_pts is not None, "Manjkajo mapped_pts"

        K = orig_pts.shape[0]

        UU = np.zeros((K,K), dtype=float)

        for i in range(K):
            UU[i,:] = getRadialValues(orig_pts[i,:], orig_pts)

        oP["alphas"] = np.linalg.solve(UU, mapped_pts[:,0])
        oP["betas"] = np.linalg.solve(UU, mapped_pts[:,1])
        oP["pts"] = orig_pts
        
    return oP

if __name__ == "__main__":
    T = getParameters("affine", rot=30)
    print(T)

# Naloga 3:
def getRadialValues(iXY, iCP):
    K = iCP.shape[0]

    # instanciranje izhodnih radialnih uteži
    oValue = np.zeros(K)

    x_i, y_i = iXY
    for k in range(K):
        x_k, y_k = iCP[k]

        # razdalja vhodne tocke do k-te kontrolne točke
        r = np.sqrt((x_i - x_k) ** 2 + (y_i -y_k) ** 2)

        # apliciranje radialne funkcije na r
        if r > 0:
            oValue[k] = -(r**2) * np.log(r)

    return oValue

def transformImage(iType, iImage, iDim, iP, iInterp, iBgr=0):
    Y, X = iImage.shape
    dx, dy = iDim

    oImage = np.ones((Y,X))*iBgr

    for y in range(Y):
        for x in range(X):
            x_hat, y_hat = x*dx, y*dy
            
            if iType == "affine":
                x_hat, y_hat, _ = iP @ np.array([x_hat,y_hat,1])

            if iType == "radial":
                U = getRadialValues([x_hat,y_hat], iP["pts"])
                x_hat,y_hat = np.array([U @ iP["alphas"], U @ iP["betas"]])            

            x_hat, y_hat = x_hat/dx, y_hat/dy

            if iInterp == 0:
                x_hat, y_hat = round(x_hat), round(y_hat)
                if 0 <= x_hat < X and 0 <= y_hat < Y:
                    oImage[y, x] = iImage[y_hat,x_hat]    
    return oImage


if __name__ == "__main__":
    imSize = [256,512]
    pxDim = [2,1]

    gX = np.arange(imSize[0])*pxDim[0]
    gY = np.arange(imSize[1])*pxDim[1]

    T = getParameters("affine", rot=30)
    print(T)

    brg = 63

    tImage = transformImage("affine", I, pxDim, np.linalg.inv(T), iInterp = 0, iBgr=brg)
    displayImage(tImage, "Affina preslikava", gX,gY)

    xy = np.array([[0,0],[511,0],[0,511],[511,511]])
    uv = np.array([[0,0],[511,0],[0,511],[255,255]])
    P = getParameters("radial", orig_pts=xy, mapped_pts=uv)
    print(P)
    rImage = transformImage("radial", I, pxDim, P, iInterp = 0, iBgr=brg)
    displayImage(rImage, "Radialna preslikava", gX,gY)

# to do: uredi celo kodo, 2 maina, osv lib, jupyter notebook