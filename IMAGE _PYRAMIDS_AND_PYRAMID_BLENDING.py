from pydoc import cli

import numpy as np
import imageio
from skimage.color import rgb2gray
from scipy.ndimage import filters
from scipy import signal
import os

import matplotlib.pyplot as plt

MAX_PIXEL_VALUE = 255
RGB_DIM = 3
GRAYSCALE = 1
RGB = 2


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    im = imageio.imread(filename)
    if np.ndim(im) == RGB_DIM and representation == GRAYSCALE:
        im = rgb2gray(im)
    im_float = im.astype(np.float64)
    if np.amax(im_float) > 1:
        im_float /= MAX_PIXEL_VALUE

    return im_float


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blured = filters.convolve(im, blur_filter)
    blured = filters.convolve(blured, np.transpose(np.array(blur_filter)))

    return np.transpose(np.transpose(blured[::2])[::2])


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    expanded = np.zeros((len(im) * 2, len(im[0]) * 2), dtype=np.float64)
    expanded[0::2, 0::2] = im

    blured_expanded = filters.convolve(expanded, np.array(blur_filter) * 2)
    return filters.convolve(blured_expanded, np.transpose(np.array(blur_filter)) * 2)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """

    cur_im = im
    pyr = [im]
    blur_filter = [1, 1]
    for _ in range(filter_size - 2):
        blur_filter = signal.convolve(blur_filter, np.array([1, 1]))
    blur_filter = blur_filter / np.sum(blur_filter)

    x_resolution = len(im)
    y_resolution = len(im[0])
    while x_resolution // 2 >= 16 and y_resolution // 2 >= 16 and max_levels - 1 >= 1:
        cur_im = reduce(cur_im, [blur_filter])
        pyr.append(cur_im)
        x_resolution = len(cur_im)
        y_resolution = len(cur_im[0])
        max_levels -= 1
    return pyr, np.array([blur_filter])


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gaussian_pyr, blur_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        min_width = min(len(gaussian_pyr[i]), len(expand(gaussian_pyr[i + 1], blur_filter)))
        min_length = min(len(gaussian_pyr[i][0]), len(expand(gaussian_pyr[i + 1], blur_filter)[0]))
        laplacian_pyr.append(np.subtract(gaussian_pyr[i][:min_width, :min_length],
                                         expand(gaussian_pyr[i + 1], blur_filter)[:min_width, :min_length]))

    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr, blur_filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    orig_im = lpyr[-1]
    i = len(lpyr) - 1
    while i >= 1:
        min_width = min(len(expand(orig_im, filter_vec)), len(lpyr[i - 1]))
        min_length = min(len(expand(orig_im, filter_vec)[0]), len(lpyr[i - 1][0]))
        orig_im = (expand(orig_im, filter_vec)[:min_width, :min_length] * coeff[i]) + lpyr[i - 1][:min_width,
                                                                                      :min_length]
        i -= 1
    return orig_im


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    res_width = 0
    # calculate the width of res
    for i in range(levels):
        res_width += len(pyr[i][0])
    res_length = len(pyr[0])
    res = np.zeros((res_length, res_width), dtype=np.float64)
    res_x_start = 0

    streched_pyr = strech_pyramid(pyr)

    for i, img in enumerate(streched_pyr):
        if i >= levels:
            break
        res[:len(img), res_x_start:res_x_start + len(img[0])] = img

        res_x_start += len(img[0])
    return res


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    L1, filter_vec_im = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    G_m = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]

    L_out = np.copy(L1)
    for k in range(len(L1)):
        L_out[k] = G_m[k] * L1[k] + (1 - G_m[k]) * L2[k]
    res = laplacian_to_image(L_out, filter_vec_im, np.ones(len(L1)))
    return np.clip(res, 0, 1)



def strech_pyramid(pyr):
    new_pyr = pyr
    for i in range(len(pyr)):
        new_pyr[i] = strech(pyr[i])
    return new_pyr


def strech(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))
