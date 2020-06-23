import sys
from typing import List

import numpy
import numpy as np
import cv2
import scipy
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve as sconvolve
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
    kernel = [1, -1]
    ans_uv = []
    ans_xy = []
    Ix = cv2.filter2D(im1, -1, kernel)
    Iy = cv2.filter2D(im1.T, -1, kernel).T
    It = im1 - im2
    IxIx = np.multiply(Ix, Ix)
    IyIy = np.multiply(Iy, Iy)
    IyIx = np.multiply(Iy, Ix)
    IxIt = np.multiply(Ix, It)
    IyIt = np.multiply(Iy, It)

    for i in range(np.shape(im1)[0]):
        for j in range(np.shape(im1)[1]):
            sum_IxIx = sumWindow(IxIx, i, j, win_size)
            sum_IyIy = sumWindow(IyIy, i, j, win_size)
            sum_IyIx = sumWindow(IyIx, i, j, win_size)

            mat = np.array([[sum_IxIx, sum_IyIx], [sum_IyIx, sum_IyIy]])
            l1, l2 = np.sort(np.linalg.eig(mat)[0])
            if (l2 > 1 and l1/l2 < 100):
                mat_inv = np.linalg.inv(mat)
                sum_IxIt = -sumWindow(IxIt, i, j, win_size)
                sum_IyIt = -sumWindow(IyIt, i, j, win_size)
                mat2 = np.array([sum_IxIt, sum_IyIt])
                mat_uv = mat_inv*mat2
                ans_uv.append(list(mat_uv))
                ans_xy.append([i,j])

    return ans_xy,ans_uv


def get_window(x, y, win_size):
    grid = np.meshgrid(range(int(-win_size/2),(win_size//2) + 1), range(int(-win_size/2),win_size//2 + 1))
    grid[0] += x
    grid[1] += y
    return grid


def sumWindow (arr: np.ndarray, x: int, y: int, win_size: int):
    window = get_window(x,y,win_size)
    sum = np.sum(arr[window])
    return sum


#def blur(img, kernel):
    #return cv2.filter2D(img, -1, kernel)


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


# list = [img]
    # kernel = cv2.getGaussianKernel(5,0)
    #
    # for i in range(levels):
    #     temp = blur(list[-1], kernel)
    #     temp = subsample(temp,2)
    #     temp = blur(temp.T, kernel)
    #     temp = subsample(temp, 2)
    #     list.append(temp.T)
    #
    # return list

#not mine
def get_filter(filter_size):
    if filter_size == SHORTEST_FILTER:
        return np.expand_dims(np.ones(SHORTEST_FILTER), ROWS)

    ones = np.ones(2)
    filter_vec = ones
    for i in range(filter_size - 2):
        filter_vec = np.convolve(filter_vec, ones)
    normed_filter = filter_vec / np.sum(filter_vec)
    return np.expand_dims(normed_filter, ROWS)

#not mine
def reduce(im, filter_vec):
    # filter_vec = np.zeros(shape=(130, 130), dtype=np.float32)
    # filter_vec = filter_vec[:, :, None]
    blurred = blur_gus(im, filter_vec)
    return blurred[::2, ::2]

#not mine
def blur_gus(im, filter_vec):
    #filter_vec = np.reshape(filter_vec, (130,130))
    filter_vec = filter_vec[:, :, None]
    row_blurred = sconvolve(im, filter_vec, mode='nearest')
    col_blurred = sconvolve(np.transpose(row_blurred), filter_vec)
    return np.transpose(col_blurred)

def blur_lup(im, filter_vec):
    #filter_vec = np.reshape(filter_vec, (130,130))
    filter_vec = filter_vec[:, :, None]
    row_blurred = sconvolve(im, filter_vec, mode='nearest')
    col_blurred = sconvolve(np.transpose(row_blurred), filter_vec)
    return np.transpose(col_blurred)


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    arr = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    arr[0:-1:2][0:-1:2] = np.copy(img)
    arr = blur_gus(arr, gs_k)
    return arr


#Creates a Laplacian pyramid
def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    filter_vec = get_filter(10)
    g_pyr = gaussianPyr(img, levels)
    pyr = []
    for i in range(len(g_pyr) - 1):
        expanded = expand(g_pyr[i + 1], g_pyr[i].shape, filter_vec)
        pyr.append(g_pyr[i] - expanded)
    pyr.append(g_pyr[SMALLEST_PYR])
    return pyr


def expand(im, shape, filter_vec):
    zeros = np.zeros(shape)
    zeros[::2, ::2] += im
    return blur_lup(zeros, 2*filter_vec)

#Return the original image from a laplacian pyramid
def laplaceianExpand(lap_pyr: List[np.ndarray], coeff, filter_vec) -> np.ndarray:
    lpyr = [lap_pyr[i] * coeff[i] for i in range(len(lap_pyr))]
    for i in range(len(lpyr) - 2, -1, -1):
        lpyr[i] = lpyr[i] + expand_im(lpyr[i + 1], filter_vec)
    return lpyr[i]

def expand_im(im, g_kernel):
    """
    Pad the images with zeroes
    :param im: image to expand
    :param g_kernel: to blur with
    :return: the expanded image
    """
    padded_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2, im.shape[2]), dtype=float)
    padded_im[::2, ::2] = im.copy()
    toReturn = blur_lup(padded_im, g_kernel * 2)
    return toReturn

def blur_image(im, g_filter):
    """
    Blur the given image using convolution with the given filter
    :param im: to blur
    :param g_filter: to be used in convolution
    :return: the blurred image
    """
    # blur at one dimension at a time for performance optimization
    g_filter = g_filter[:, :, None]
    convolved_im = scipy.ndimage.filters.convolve(im, g_filter, mode='reflect')
    convolved_im = scipy.ndimage.filters.convolve(convolved_im, g_filter.transpose(), mode='reflect')
    return convolved_im

def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    lapl_pyr1 = laplaceianReduce(img_1, levels)
    lapl_pyr2 = laplaceianReduce(img_2, levels)
    float_mask = mask.astype(float)
    mask_pyr = gaussianPyr(float_mask, levels)
    Lout = [0] * len(lapl_pyr1)
    for i in range(len(lapl_pyr1)):
        Lout[i] = np.multiply(lapl_pyr1[i], mask_pyr[i]) + np.multiply(1 - mask_pyr[i], lapl_pyr2[i])
    coeff = [1] * len(lapl_pyr2)
    filter_vec = get_filter(10)
    blend_im = laplaceianExpand(Lout, coeff, filter_vec)
    return blend_im.clip(0, 1)
