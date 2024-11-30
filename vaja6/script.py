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
    oValue = np.zeros(K)
    x_i, y_i = iXY

    for k in range(K):
        x_k, y_k = iCP[k]

        r = np.sqrt((x_i - x_k) ** 2 + (y_i -y_k) ** 2)

        if r > 0:
            oValue[k] = -(r**2) * np.log(r)

    return oValue

def transformImage(iType, iImage, iDim, iP, iBgr=0, iInterp=0):
    Y, X = iImage.shape
    oImage = np.ones((Y, X)) * iBgr
    dx, dy = iDim

    for y in range(Y):
        for x in range(X):
            x_hat, y_hat = x * dx, y * dy

            if iType == "affine":
                x_hat, y_hat, _ = iP @ np.array([x_hat, y_hat, 1])

            if iType == "radial":
                U = getRadialValues([x_hat, y_hat], iP["pts"])
                x_hat, y_hat = np.array([U @ iP["alphas"], U @ iP["betas"]])

            x_hat, y_hat = x_hat / dx, y_hat / dy

            if iInterp == 0:  # interpolacija 0. reda
                x_hat, y_hat = round(x_hat), round(y_hat)

                if 0 <= x_hat < X and 0 <= y_hat < Y:
                    oImage[y, x] = iImage[y_hat, x_hat]

            if iInterp == 1:  # interpolacija 1. reda

                x0 = int(np.floor(x_hat))
                y0 = int(np.floor(y_hat))
                x1 = x0 + 1
                y1 = y0 + 1

                if 0 <= x0 < X and 0 <= y0 < Y and 0 <= x1 < X and 0 <= y1 < Y:
                    a = abs(x_hat - x0) * abs(y_hat - y1)
                    b = abs(x_hat - x1) * abs(y_hat - y1)
                    c = abs(x_hat - x1) * abs(y_hat - y0)
                    d = abs(x_hat - x0) * abs(y_hat - y0)

                    s = (
                        a * iImage[y0, x1]
                        + b * iImage[y0, x0]
                        + c * iImage[y1, x0]
                        + d * iImage[y1, x1]
                    )

                    oImage[y, x] = int(s)
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

# Dodatno: Naloga 1
if __name__ == "__main__":
    imSize = [256, 512]
    pxDim = [2, 1]
    bgr = 63

    gx = np.arange(imSize[0]) * pxDim[0]
    gy = np.arange(imSize[1]) * pxDim[1]

    image = loadImage("vaja6/data/grid-256x512-08bit.raw", imSize, np.uint8)

    # paramteri preslikave: gxy = 0.5 (shear), ky = 0.8 (scale)

    T = getParameters("affine", scale=[1, 0.8], shear=[0.5, 0])
    print(T)

    Timage1 = transformImage("affine", image, pxDim, np.linalg.inv(T), bgr, iInterp=0)  # interpolacija 0. reda
    Timage2 = transformImage("affine", image, pxDim, np.linalg.inv(T), bgr, iInterp=1)  # interpolacija 1. reda

    displayImage(image, "Originalna slika", gx, gy)
    displayImage(Timage1, "Affina transformacija - interpolacija 0. reda", gx, gy)
    displayImage(Timage2, "Affina transformacija - interpolacija 1. reda", gx, gy)

    # Dodatno: Naloga 2
    image = loadImage("vaja6/data/lena-256x512-08bit.raw", imSize, np.uint8)

    # a) kx = 0.7, ky = 1.4
    T = getParameters("affine", scale=[0.7, 1.4])
    Timage = transformImage("affine", image, pxDim, np.linalg.inv(T), iBgr=bgr, iInterp=1)
    displayImage(Timage, "Affina transformacija: kx = 0.7, ky = 1.4", gx, gy)

    # b) tx = 20, ty = -10
    T = getParameters("affine", trans=[20, -10])
    Timage = transformImage("affine", image, pxDim, np.linalg.inv(T), iBgr=bgr, iInterp=1)
    displayImage(Timage, "Affina transformacija: tx = 20, ty = -10", gx, gy)

    # c) rot = 30
    T = getParameters("affine", rot=30)
    Timage = transformImage("affine", image, pxDim, np.linalg.inv(T), iBgr=bgr, iInterp=1)
    displayImage(Timage, "Affina transformacija: rot = 30", gx, gy)

    # d) gxy = 0.1, gyx = 0.5
    T = getParameters("affine", shear=[0.1, 0.5])
    Timage = transformImage("affine", image, pxDim, np.linalg.inv(T), iBgr=bgr, iInterp=1)
    displayImage(Timage, "Affina transformacija: gxy = 0.1, gyx = 0.5", gx, gy)

    # e) tx = -10, ty = 20, rot = 15
    T = getParameters("affine", trans=[-10, 20], rot=15)
    Timage = transformImage("affine", image, pxDim, np.linalg.inv(T), iBgr=bgr, iInterp=1)
    displayImage(Timage, "Affina transformacija: tx = -10, ty = 20, rot = 15", gx, gy)

    # f) kx = 0.7, ky = 0.7, tx = 30, ty = -20, rot = -15
    T = getParameters("affine", scale=[0.7, 0.7], trans=[30, -20], rot=-15)
    Timage = transformImage("affine", image, pxDim, np.linalg.inv(T), iBgr=bgr, iInterp=1)
    displayImage(Timage, "Affina transformacija: kx = 0.7, ky = 0.7, tx = 30, ty = -20, rot = -15", gx, gy)

# Dodatno: Naloga 3
# Kako se imenuje preslikava iz vprašanja 2(e)?
#
# Vprašanje 2(e) vsebuje translacijo in rotacijo kar se pravi da gre za "Togo preslikavo".
# Lastnosti toge preslikave so sledece:
#   -> ohranja vzporednost med premicami
#   -> ohranja kote med premicami
#   -> ohranja razdalje med poljubnimi tockami
#
#
# Kako se imenuje preslikava iz vprašanja 2(f)?
#
# Vprašanje 2(f) vsebuje izotropno skaliranje (v obe smeri enako skaliranje), translacijo in rotacijo kar se pravi da gre za "Podobnostno preslikavo".
# Lastnosti podobnostne preslikave so sledece:
#   -> ohranja vzporednost med premicami
#   -> ohranja kote med premicami
#   -> ne ohranja razdalje med poljubnimi tockami

# Dodatno: Naloga 4
def getParametersUpgraded(iType, scale=None, trans=None, rot=None, shear=None, orig_pts=None, mapped_pts=None, centered=False, imSize=None):

    oP = {}
    if iType == "affine":
        if scale == None:
            scale = [1, 1]

        if rot == None:
            rot = 0

        if trans == None:
            trans = [0, 0]

        if shear == None:
            shear = [0, 0]

        Tk = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]])  # scale

        Tt = np.array([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]])  # translation

        phi = rot * np.pi / 180

        Tf = np.array(
            [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
        )  # rotation

        Tg = np.array([[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]])  # shear

        Transform = Tg @ Tf @ Tt @ Tk

        if centered and imSize is not None:
            # v argumentu funkcije je potrebno postaviti "centered" na 1 in podati velikosti slike da se izvede afina transformacija okoli sredisca
            # pri podajanju velikosti slike je potrebno paziti na razmerje velikosti pixel-a

            x, y = imSize
            x_center, y_center = x / 2 - 1, y / 2 - 1

            Toffset = np.array([[1, 0, -x_center], [0, 1, -y_center], [0, 0, 1]])
            Toffset_back = np.array([[1, 0, x_center], [0, 1, y_center], [0, 0, 1]])

            oP = Toffset_back @ Transform @ Toffset

        else:
            oP = Transform

    elif iType == "radial":
        assert orig_pts is not None, "manjka orig_pts"
        assert mapped_pts is not None, "manjka mapped_pts"

        K = orig_pts.shape[0]  # stevilo kontrolnih tock
        UU = np.zeros(
            (K, K), dtype=float
        )  # insticiramo matriko radialnih funkcij kontrolnih tock

        for i in range(K):
            UU[i, :] = getRadialValues(
                orig_pts[i, :], orig_pts
            )  # izracunamo radialne funkcije za vsako kontrolno tocko do vseh ostalih

        # resimo sistem lin enacb ki nam da projekcijo kontrolnih tock do mapped_pts
        oP["alphas"] = np.linalg.lstsq(UU, mapped_pts[:, 0])[0]
        oP["betas"] = np.linalg.lstsq(UU, mapped_pts[:, 1])[0]
        oP["pts"] = orig_pts

    return oP


if __name__ == "__main__":
    imSize = [256, 512]
    pxDim = [2, 1]
    bgr = 63

    gx = np.arange(imSize[0]) * pxDim[0]
    gy = np.arange(imSize[1]) * pxDim[1]

    image = loadImage("vaja6/data/lena-256x512-08bit.raw", imSize, np.uint8)
    displayImage(image, "Original image", gx, gy)

    # parametri preslikave 2(c): rot = -30
    T = getParametersUpgraded("affine", rot=-30, centered=True, imSize=[imSize[0] * pxDim[0], imSize[1] * pxDim[1]])

    Timage = transformImage("affine", image, pxDim, np.linalg.inv(T), iBgr=bgr, iInterp=1)
    displayImage(Timage, "Affina transformacija: rot = -30", gx, gy)

    # parametri preslikave 2(d): gxy = 0.1, gyx = 0.5
    T = getParametersUpgraded("affine", shear=[0.1, 0.5], centered=True, imSize=[imSize[0] * pxDim[0], imSize[1] * pxDim[1]])

    Timage = transformImage("affine", image, pxDim, np.linalg.inv(T), iBgr=bgr, iInterp=1)
    displayImage(Timage, "Affina transformacija: gxy = 0.1, gyx = 0.5", gx, gy)


# Dodatno: Naloga 5
def displayPoints(iXY, iMarker, edgecolor="blue", facecolor="none", markersize=10, linewidth=2, labelPoints=False):
    # potrebno je bilo malo adjustanja funkcije zato da je koncni rezultat bolj podoben danemu primeru v navodilih
    # dodatno je bil dodan izpis koordinat tock oz. "label"
    if "bo" in iMarker:
        plt.scatter(iXY[:, 0], iXY[:, 1], edgecolor=edgecolor, facecolor=facecolor, s=markersize**2, linewidth=linewidth, clip_on=False)
        if labelPoints:
            for i, (x, y) in enumerate(iXY):
                plt.text(x + 5, y + 5, f"({x}, {y})", ha="left", va="bottom", fontsize=10, clip_on=False, color="blue")
    else:
        plt.plot(iXY[:, 0], iXY[:, 1], iMarker, ms=markersize * 0.8, lw=linewidth, clip_on=False)
        
        if labelPoints:
            for i, (x, y) in enumerate(iXY):
                plt.text(x + 5, y - 5, f"({x}, {y})", ha="left", va="top", fontsize=10, clip_on=False, color="red")

if __name__ == "__main__":
    imSize = [256, 512]
    pxDim = [2, 1]
    bgr = 63

    gx = np.arange(imSize[0]) * pxDim[0]
    gy = np.arange(imSize[1]) * pxDim[1]

    # kontrolne tocke
    xy = np.array(
        [
            [0, 0],
            [511, 0],
            [0, 511],
            [511, 511],
            [63, 63],
            [64, 447],
            [447, 63],
            [447, 447],
        ]
    )

    # preslikane tocke
    uv = np.array(
        [
            [0, 0],
            [511, 0],
            [0, 511],
            [511, 511],
            [127, 95],
            [127, 415],
            [383, 95],
            [383, 415],
        ]
    )

    image1 = loadImage("vaja6/data/lena-256x512-08bit.raw", imSize, np.uint8)
    image2 = loadImage("vaja6/data/grid-256x512-08bit.raw", imSize, np.uint8)
    P = getParameters("radial", orig_pts=xy, mapped_pts=uv)

    rImage1 = transformImage("radial", image1, pxDim, P, iBgr=bgr, iInterp=0)
    rImage2 = transformImage("radial", image2, pxDim, P, iBgr=bgr, iInterp=0)

    # "Popravljena" transformacija (glej spodaj komentar)
    P_fixed = getParameters("radial", orig_pts=uv, mapped_pts=xy)
    rImage1_fixed = transformImage("radial", image1, pxDim, P_fixed, iBgr=bgr, iInterp=0)
    rImage2_fixed = transformImage("radial", image2, pxDim, P_fixed, iBgr=bgr, iInterp=0)

    displayImage(rImage1, "Radialna transformacija", gx, gy)
    displayPoints(uv, "bo", labelPoints=True)
    displayPoints(xy, "rx", labelPoints=True)

    displayImage(image1, "Original image", gx, gy)
    displayPoints(xy, "rx", labelPoints=True)

    displayImage(rImage2, "Radialna transformacija", gx, gy)
    displayPoints(uv, "bo")
    displayPoints(xy, "rx")

    displayImage(image2, "Original image", gx, gy)
    displayPoints(xy, "rx")

    displayImage(rImage1_fixed, "Radialna transformacija - popravljena", gx, gy)

    displayImage(rImage2_fixed, "Radialna transformacija - popravljena", gx, gy)

# Ali glede na polozaj tock preslikava deluje pravilno?
#
# Pomoje dela pravilno ¯\_(ツ)_/¯
#
# X -> kontrolne tocke, O -> preslikane tocke
#
# Ce na sliki naredimo izpis koordinat originalnih in preslikanih kontrolnih tock tock, se vidi, da je preslikava uspesno preslikana tocke.
#
# Komentar asistenta:
# Pri zadnji nalogi bi lahko pričakovali da bi se slika radialno skrčila proti središčnim tarčnim kontrolnim točkam torej od xy -> uv.
# Če pa pogledamo rezultat, pa se zgodi obratno; slika se raztegne od uv -> xy. Torej za pravilno transformacijo moramo uv, in xy seta kontrolnih točk zamenjati.