from typing import List
import numpy as np
import cv2
import scipy
from scipy.ndimage.filters import convolve as sconvolve
from numpy import linalg as LA
import matplotlib.pyplot as plt

ROWS = 0
SHORTEST_FILTER = 1
MIN_DIM = 16
COLS = 1


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    xy_ans = []
    uv_ans = []
    w = int(win_size / 2)

    if isgray(im1) == False:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if isgray(im2) == False:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    Ix = cv2.filter2D(im2, -1, x)
    Ix = cv2.GaussianBlur(Ix, (5, 5), 0)
    Iy = cv2.filter2D(im2, -1, y)
    Iy = cv2.GaussianBlur(Iy, (5, 5), 0)
    It = im2 - im1

    for i in range(w, im1.shape[0] - w, step_size):
        for j in range(w, im1.shape[1] - w, step_size):
            ix = Ix[i - w:i + w + 1, j - w:j + w + 1].flatten()
            iy = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            it = It[i - w:i + w + 1, j - w:j + w + 1].flatten()
            a2 = it.reshape(it.shape[0], 1)*(-1)
            a1 = np.full((win_size ** 2, 2), 0)
            a1[:, 0] = ix
            a1[:, 1] = iy
            At = np.transpose(a1)
            ATA = At.dot(a1)
            eg, v = LA.eig(ATA)
            eg = np.sort(eg)
            l1 = eg[-1]
            l2 = eg[-2]

            if (l1 / l2) < 100 and (l2 > 1):
                temp = At.dot(a2)
                ATAr = LA.inv(ATA)
                v = ATAr.dot(temp)
                uv_ans.append(v)
                xy_ans.append([j, i])

    xy_ans = np.asarray(xy_ans)
    uv_ans = np.asarray(uv_ans)
    return xy_ans, uv_ans


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    pyrLst = [img]
    gauss = cv2.getGaussianKernel(5, sigma=(0.3+0.8))
    gauss = np.outer(gauss, gauss.transpose())
    for i in range(1, levels):
        Imgt = cv2.filter2D(pyrLst[i-1], -1, gauss)
        Imgt = Imgt[:: 2, :: 2]
        pyrLst.append(Imgt)
    return pyrLst


def reduce(im, filter_vec):
    blurred = blur(im, filter_vec)
    return blurred[::2, ::2]


def blur(im, filter_vec):
    if (np.size(im.shape) == 3):
        filter_vec = filter_vec[:, :, None]
    row_blurred = sconvolve(im, filter_vec, mode='nearest')
    col_blurred = sconvolve(np.transpose(row_blurred), filter_vec)
    return np.transpose(col_blurred)


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    gs_k = (gs_k * 4) / gs_k.sum()

    if np.size(img.shape) == 2:
        w, h = img.shape
        imgNew = np.full((2*w, 2*h), 0, dtype=img.dtype).astype(np.float)
        imgNew[::2, ::2] = img

    elif np.size(img.shape) == 3:
        w, h, t = img.shape
        imgNew = np.full((2*w, 2*h, t), 0, dtype=img.dtype).astype(np.float)
        imgNew[::2, ::2] = img

    imgNew = cv2.filter2D(imgNew, -1, gs_k, borderType=cv2.BORDER_DEFAULT)
    return imgNew


#Creates a Laplacian pyramid
def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    ans = []
    gauss = gaussianPyr(img, levels)
    tmp = 0

    for i in range(len(gauss) - 1):
        s = gauss[i + 1]
        exp = gaussExpand(s, gaussKernel)
        if gauss[i].shape == exp.shape:
            tmp = gauss[i].shape
        if tmp == 0:
            exp = exp[:-1, :-1]
        level = gauss[i] - exp
        ans.append(level)

    ans.append(gauss[-1])
    return ans


#Return the original image from a laplacian pyramid
def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    lap_list = lap_pyr[::-1]
    ans = lap_list[0]

    for i in range(len(lap_list) - 1):
        exp = gaussExpand(ans, gaussKernel)
        a = lap_list[i + 1].shape == exp.shape
        if a == 0:
            exp = exp[:-1, :-1]
        ans = (lap_list[i + 1] + exp)
    return ans


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    mat = []
    Blendimg = img_1 * mask + img_2 * (1 - mask)
    lapl_pyr1 = laplaceianReduce(img_1, levels)
    lapl_pyr2 = laplaceianReduce(img_2, levels)
    float_mask = mask.astype(float)
    mask_pyr = gaussianPyr(float_mask, levels)
    Lout = [0] * len(lapl_pyr1)

    for i in range(len(lapl_pyr1)):
        Lout[i] = np.multiply(lapl_pyr1[i], mask_pyr[i]) + np.multiply(1 - mask_pyr[i], lapl_pyr2[i])
    for i in range(levels):
        mat.append(mask_pyr[i] * lapl_pyr1[i] + (1 - mask_pyr[i]) * lapl_pyr2[i])

    Blendedimg2 = laplaceianExpand(mat)
    return Blendimg, Blendedimg2


def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2] == 1: return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all(): return True
    return False

gaussKernel = np.array([[1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445],
                        [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                        [6.49510362, 25.90969361, 41.0435344, 25.90969361, 6.49510362],
                        [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                        [1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445]])

