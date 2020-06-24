from typing import List
import numpy as np
import cv2
import scipy
from scipy.ndimage.filters import convolve as sconvolve
from numpy import linalg as LA
import matplotlib.pyplot as plt

MIN_DIM = 16
COLS = 1
ROWS = 0
LARGEST_PYR = 0
SMALLEST_PYR = -1
REVERSED = -1
SHORTEST_FILTER = 1
PYR = 0
NORMALIZE_FACTOR = 255
GRAYSCALE = 1
RGB = 2
RED = 0
GREEN = 1


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    xy_ans = []
    uv_ans = []

    if (not isgray(im1)):
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if (not isgray(im2)):
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    Ix = cv2.filter2D(im2, -1, kernelx)
    Ix = cv2.GaussianBlur(Ix, (5, 5), 0)
    Iy = cv2.filter2D(im2, -1, kernely)
    Iy = cv2.GaussianBlur(Iy, (5, 5), 0)
    It = im2 - im1

    w = int(win_size / 2)
    for i in range(w, im1.shape[0] - w, step_size):
        for j in range(w, im1.shape[1] - w, step_size):
            ix = Ix[i - w:i + w + 1, j - w:j + w + 1].flatten()
            iy = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            it = It[i - w:i + w + 1, j - w:j + w + 1].flatten()
            B = it.reshape(it.shape[0], 1) * (-1)
            A = np.full((win_size ** 2, 2), 0)
            A[:, 0] = ix
            A[:, 1] = iy
            At = np.transpose(A)
            ATA = At.dot(A)

            eginvalue, v = LA.eig(ATA)
            eginvalue = np.sort(eginvalue)
            l1 = eginvalue[-1]
            l2 = eginvalue[-2]

            if (l2 > 1) and (l1 / l2) < 100:
                ATAreverse = LA.inv(ATA)
                z = At.dot(B)
                v = ATAreverse.dot(z)
                xy_ans.append([j, i])
                uv_ans.append(v)
    xy_ans = np.asarray(xy_ans)
    uv_ans = np.asarray(uv_ans)
    return (xy_ans, uv_ans)

def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2] == 1: return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all(): return True
    return False


def subsample(img, param):
    return img[0:-1:param]


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    filter_vec = get_filter(10)
    pyr = [img]
    for i in range(levels - 1):
        if pyr[i].shape[ROWS] // 2 <= MIN_DIM or pyr[i].shape[COLS] // 2 <= MIN_DIM:
            break
        pyr.append(reduce(pyr[i], filter_vec))

    return pyr


def get_filter(filter_size):
    if filter_size == SHORTEST_FILTER:
        return np.expand_dims(np.ones(SHORTEST_FILTER), ROWS)
    ones = np.ones(2)
    filter_vec = ones
    for i in range(filter_size - 2):
        filter_vec = np.convolve(filter_vec, ones)
    normed_filter = filter_vec / np.sum(filter_vec)
    return np.expand_dims(normed_filter, ROWS)


def reduce(im, filter_vec):
    blurred = blur(im, filter_vec)
    return blurred[::2, ::2]


def blur(im, filter_vec):
    if (np.size(im.shape) == 3):
        filter_vec = filter_vec[:, :, None]
    row_blurred = sconvolve(im, filter_vec, mode='nearest')
    col_blurred = sconvolve(np.transpose(row_blurred), filter_vec)
    return np.transpose(col_blurred)


def gaussExpand(img: np.ndarray, gs: np.ndarray) -> np.ndarray:
    if (len(img.shape) == 2):
        w, h = img.shape
        imgNew = np.full((2 * w, 2 * h), 0, dtype=img.dtype)
        imgNew = imgNew.astype(np.float)
        imgNew[::2, ::2] = img

    if (len(img.shape) == 3):
        w, h, z = img.shape
        imgNew = np.full((2 * w, 2 * h, z), 0, dtype=img.dtype)
        imgNew = imgNew.astype(np.float)
        imgNew[::2, ::2] = img

    gs = (gs * 4) / gs.sum()
    ans = cv2.filter2D(imgNew, -1, gs, borderType=cv2.BORDER_DEFAULT)

    return ans


#Creates a Laplacian pyramid
def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    gaussList = gaussianPyr(img, levels)
    ans = []

    for i in range(len(gaussList) - 1):
        s = gaussList[i + 1]
        exp = gaussExpand(s, gaussKernel)
        a = gaussList[i].shape == exp.shape
        if (not a): exp = exp[:-1, :-1]
        newLevel = gaussList[i] - exp
        ans.append(newLevel)
    ans.append(gaussList[-1])
    
    return ans


#Return the original image from a laplacian pyramid
def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    lap_list = lap_pyr[::-1]
    ans = lap_list[0]
    length = len(lap_list)

    for i in range(length - 1):
        exp = gaussExpand(ans, gaussKernel)
        a = lap_list[i + 1].shape == exp.shape
        if (not a): exp = exp[:-1, :-1]
        ans = (lap_list[i + 1] + exp)
    return ans


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    Blendimg = img_1 * mask + img_2 * (1 - mask)
    lapl_pyr1 = laplaceianReduce(img_1, levels)
    lapl_pyr2 = laplaceianReduce(img_2, levels)
    mat = []
    float_mask = mask.astype(float)
    mask_pyr = gaussianPyr(float_mask, levels)
    Lout = [0] * len(lapl_pyr1)
    for i in range(len(lapl_pyr1)):
        Lout[i] = np.multiply(lapl_pyr1[i], mask_pyr[i]) + np.multiply(1 - mask_pyr[i], lapl_pyr2[i])
    for i in range(levels):
        mat.append(mask_pyr[i] * lapl_pyr1[i] + (1 - mask_pyr[i]) * lapl_pyr2[i])
    Blendedimg2 = laplaceianExpand(mat)

    return Blendimg, Blendedimg2


gaussKernel = np.array([[1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445],
                        [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                        [6.49510362, 25.90969361, 41.0435344, 25.90969361, 6.49510362],
                        [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                        [1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445]])
