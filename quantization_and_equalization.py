
import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

MAX_PIXEL_VALUE=255
RGB_DIM = 3
GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


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


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    im = read_image(filename, representation)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    r = imRGB[:, :, 0]
    g = imRGB[:, :, 1]
    b = imRGB[:, :, 2]

    y = r * RGB_YIQ_TRANSFORMATION_MATRIX[0, 0] + g * RGB_YIQ_TRANSFORMATION_MATRIX[0, 1] + b * \
        RGB_YIQ_TRANSFORMATION_MATRIX[0, 2]
    i = r * RGB_YIQ_TRANSFORMATION_MATRIX[1, 0] + g * RGB_YIQ_TRANSFORMATION_MATRIX[1, 1] + b * \
        RGB_YIQ_TRANSFORMATION_MATRIX[1, 2]
    q = r * RGB_YIQ_TRANSFORMATION_MATRIX[2, 0] + g * RGB_YIQ_TRANSFORMATION_MATRIX[2, 1] + b * \
        RGB_YIQ_TRANSFORMATION_MATRIX[2, 2]

    imYIQ = imRGB.copy()

    imYIQ[:, :, 0] = y
    imYIQ[:, :, 1] = i
    imYIQ[:, :, 2] = q

    return imYIQ


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    yiq_rgb_transformation_matrix = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)

    y = imYIQ[:, :, 0]
    i = imYIQ[:, :, 1]
    q = imYIQ[:, :, 2]

    r = y * yiq_rgb_transformation_matrix[0, 0] + i * yiq_rgb_transformation_matrix[0, 1] + q * \
        yiq_rgb_transformation_matrix[0, 2]
    g = y * yiq_rgb_transformation_matrix[1, 0] + i * yiq_rgb_transformation_matrix[1, 1] + q * \
        yiq_rgb_transformation_matrix[1, 2]
    b = y * yiq_rgb_transformation_matrix[2, 0] + i * yiq_rgb_transformation_matrix[2, 1] + q * \
        yiq_rgb_transformation_matrix[2, 2]

    imRGB = imYIQ.copy()

    imRGB[:, :, 0] = r
    imRGB[:, :, 1] = g
    imRGB[:, :, 2] = b

    return imRGB


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    im_tmp = im_orig
    im_tmp_3_channels = im_orig
    if np.ndim(im_orig) == 3:
        im_tmp_3_channels = rgb2yiq(im_orig)  # converting the RGB image to YIQ.
        im_tmp = im_tmp_3_channels[:, :, 0]  # taking only the Y channel of the converted image.

    # performing histogram equalization algorithm as seen in class
    hist_orig = np.histogram(im_tmp, bins=256, range=(0, 1))[0]
    hist_sum = np.cumsum(hist_orig)
    sum_pixels = np.sum(hist_orig)
    first_none_zero_index = (hist_sum != 0).argmax(axis=0)
    first_none_zero_value = hist_sum[first_none_zero_index]
    if hist_sum[MAX_PIXEL_VALUE] - first_none_zero_value == 0:
        return [im_orig, hist_orig, hist_orig]
    hist_new_indexes = (MAX_PIXEL_VALUE * (
            (hist_sum - first_none_zero_value) / (sum_pixels - first_none_zero_value))).round()
    hist_new_indexes = hist_new_indexes.astype("int64")
    hist_eq = np.zeros(256, dtype=np.float64)
    np.put(hist_eq, hist_new_indexes, hist_orig)

    # replace value in im_orig with hist_new_indexes[value]
    mapping = np.zeros(256, dtype=np.float64)
    mapping[np.arange(256)] = hist_new_indexes
    im_tmp *= MAX_PIXEL_VALUE
    im_tmp = im_tmp.astype("int64")
    im_eq = mapping[im_tmp]
    im_eq = im_eq.astype(np.float64)
    im_eq /= MAX_PIXEL_VALUE

    if np.ndim(im_orig) == 3:
        im_tmp_3_channels[:, :, 0] = im_eq
        im_eq = yiq2rgb(im_tmp_3_channels)

    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    im_tmp = im_orig
    im_tmp_3_channels = im_orig

    if np.ndim(im_orig) == 3:
        im_tmp_3_channels = rgb2yiq(im_orig)  # converting the RGB image to YIQ.
        im_tmp = im_tmp_3_channels[:, :, 0]  # taking only the Y channel of the converted image.

    hist = np.histogram(im_tmp, bins=256, range=(0, 1))[0]  # create image histogram
    hist_cum = np.cumsum(hist)  # create cumulative histogram
    z_arr = np.zeros(n_quant + 1, dtype=np.int64)  # computing z of size shape (n_quant+1,)
    z_arr[0] = -1
    z_arr[-1] = MAX_PIXEL_VALUE
    sum_pixels = hist_cum[-1]
    bin_pixels_avg = sum_pixels / n_quant  # initial amount of pixels in each bin of z
    q_arr = np.zeros(n_quant, dtype=np.int64)
    for i in range(1, n_quant):
        z_arr[i] = np.argmax(hist_cum >= (bin_pixels_avg * i))

    # initialize q
    for i in range(n_quant):
        q_arr[i] = (z_arr[i + 1] + z_arr[i]) // 2

    z_error = []
    error = []
    iter_runner = 0
    convergence = False
    indx_arr = np.arange(256)
    while (iter_runner < n_iter) and (not convergence):
        error_count = 0
        for q_index in range(n_quant):
            # computing z_i

            if q_index != 0:
                new_z = (q_arr[q_index - 1] + q_arr[q_index]) // 2
                if z_arr[q_index] != new_z:
                    z_arr[q_index] = int(new_z)
                    error_count += 1

        # calculating qi according to formula shown in class
        for q_index in range(n_quant):
            start = int(z_arr[q_index]) + 1
            end = int(z_arr[q_index + 1])
            new_qi_sum_top = np.sum(indx_arr[start:end + 1] * hist[start:end + 1])
            new_qi_sum_bottom = np.sum(hist[start:end + 1])
            if new_qi_sum_bottom != 0:
                q_arr[q_index] = int(new_qi_sum_top / new_qi_sum_bottom)

        # calculatin the new error according to formuale shown in class
        new_error = 0
        for q_index in range(n_quant):
            start = int(z_arr[q_index]) + 1
            end = int(z_arr[q_index + 1])
            new_error += np.sum(((q_arr[q_index] - indx_arr[start:end + 1]) ** 2) * hist[start:end + 1])
        iter_runner += 1

        error.append(new_error)
        z_error.append(error_count)
        if error_count == 0:
            convergence = True  # exit loop

    im_quant = np.zeros(im_tmp.shape, dtype=np.float64)
    im_tmp *= MAX_PIXEL_VALUE
    im_tmp = im_tmp.astype(np.int64)
    new_idx_arr = np.zeros(256, dtype=np.int64)
    for q_index in range(n_quant):
        new_idx_arr[z_arr[q_index] + 1:z_arr[q_index + 1] + 1] = q_arr[q_index]
    new_idx_arr[z_arr[-2]:z_arr[-1] + 1] = q_arr[-1]
    im_quant = new_idx_arr[im_tmp].astype(np.float64)

    im_quant /= MAX_PIXEL_VALUE

    if np.ndim(im_orig) == 3:
        im_tmp_3_channels[:, :, 0] = im_quant
        im_quant = yiq2rgb(im_tmp_3_channels)
    return [im_quant, error]