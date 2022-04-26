import numpy as np
import cv2 as cv
from scipy.interpolate import UnivariateSpline


def orb_features(filename):
    image = cv.imread(filename)
    orb = cv.ORB_create()
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    image = cv.drawKeypoints(image, kp, None, color=(255, 0, 0), flags=0)
    cv.imwrite('orb_features_' + filename, image)


def sift_features(filename):
    image = cv.imread(filename)
    sift = cv.SIFT_create()
    kp = sift.detect(image, None)
    image = cv.drawKeypoints(image, kp, image)
    cv.imwrite('sift_features_' + filename, image)


def canny_edges(filename):
    img = cv.imread(filename)
    edges = cv.Canny(img, 100, 200)
    cv.imwrite('canny_edges_' + filename, edges)


def grayscale(filename):
    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    cv.imwrite('grayscale_' + filename, image)


def hsv(filename):
    image = cv.imread(filename)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imwrite('hsv_' + filename, hsv)


def flip_vertical(filename):
    image = cv.imread(filename)
    flip = cv.flip(image, 0)
    cv.imwrite('flip_vertical_' + filename, flip)


def flip_horizontal(filename):
    image = cv.imread(filename)
    flip = cv.flip(image, 1)
    cv.imwrite('flip_horizontal_' + filename, flip)


def rotate_45_degree(filename):
    image = cv.imread(filename)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, 45, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1])
    cv.imwrite('rotate_45_degree_' + filename, result)


def rotate_30_degree(filename, x, y):
    image = cv.imread(filename)
    dot = x, y
    rot_mat = cv.getRotationMatrix2D(dot, 30, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1])
    cv.imwrite('rotate_30_degree_' + filename, result)


def shift_10_pixels(filename):
    image = cv.imread(filename)
    mtx = np.float32([[1, 0, 10], [0, 1, 0]])
    res = cv.warpAffine(image, mtx, image.shape[1::-1])
    cv.imwrite('shift_10_pixels_' + filename, res)


def change_brightness(filename, brightness):
    image = cv.imread(filename)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    if brightness > 0:
        brightness = np.uint8(brightness)
        lim = 255 - brightness
        v[v > lim] = 255
        v[v <= lim] += brightness
    if brightness < 0:
        brightness = np.uint8(-brightness)
        lim = brightness
        v[v < lim] = 0
        v[v >= lim] -= brightness

    final_hsv = cv.merge((h, s, v))
    image = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    cv.imwrite('change_brightness_' + filename, image)


def change_contrast(filename, contrast):
    image = cv.imread(filename)
    contrast *= 2
    f = 131 * (contrast + 127) / (127 * (131 - contrast))
    alpha = f
    gamma = 127 * (1 - f)
    image = cv.addWeighted(image, alpha, image, 0, gamma)
    cv.imwrite('change_contrast_' + filename, image)


def gamma_correction(filename, gamma):
    image = cv.imread(filename)

    deg = 1 / gamma
    table = [pow(i / 255, deg) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    image = cv.LUT(image, table)
    cv.imwrite('gamma_correction_' + filename, image)


def equalize_histogram(filename):
    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    image = cv.equalizeHist(image)
    cv.imwrite('equalize_histogram_' + filename, image)


def warming_filter(filename):
    image = cv.imread(filename)

    def create_LUT(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    incr = create_LUT([0, 64, 128, 192, 256], [0, 70, 140, 210, 256]).astype(np.uint8)
    decr = create_LUT([0, 64, 128, 192, 256], [0, 30, 80, 120, 192]).astype(np.uint8)

    b, g, r = cv.split(image)
    r = cv.LUT(r, incr).astype(np.uint8)
    b = cv.LUT(b, decr).astype(np.uint8)
    image = cv.merge((b, g, r))

    h, s, v = cv.split(cv.cvtColor(image, cv.COLOR_RGB2HSV))
    s = cv.LUT(s, incr).astype(np.uint8)
    image = cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2RGB)

    cv.imwrite('warming_filter_' + filename, image)


def cooling_filter(filename):
    image = cv.imread(filename)

    def create_LUT(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    incr = create_LUT([0, 64, 128, 192, 256], [0, 70, 140, 210, 256]).astype(np.uint8)
    decr = create_LUT([0, 64, 128, 192, 256], [0, 30, 80, 120, 192]).astype(np.uint8)

    b, g, r = cv.split(image)
    b = cv.LUT(r, incr).astype(np.uint8)
    r = cv.LUT(b, decr).astype(np.uint8)
    image = cv.merge((b, g, r))

    h, s, v = cv.split(cv.cvtColor(image, cv.COLOR_RGB2HSV))
    s = cv.LUT(s, decr).astype(np.uint8)
    image = cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2RGB)

    cv.imwrite('cooling_filter_' + filename, image)


def colormap(filename, template_filename):
    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    template = cv.imread(template_filename)

    b, g, r = cv.split(template)
    r = cv.LUT(image, r)
    g = cv.LUT(image, g)
    b = cv.LUT(image, b)
    image = cv.merge((b, g, r))

    cv.imwrite('colormap_' + filename, image)


def binarize(filename):
    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    _, image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    cv.imwrite('binarize_' + filename, image)


def find_contours_binarized(filename):
    image = cv.imread(filename)
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image = cv.drawContours(image, contours, -1, (255, 0, 0), 3)
    cv.imwrite('find_contours_binarized_' + filename, image)


def find_contours(filename):
    image = cv.imread(filename)

    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    image = cv.GaussianBlur(image, (3, 3), 0)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv.imwrite('find_contours_' + filename, grad)


def blur(filename):
    image = cv.imread(filename)
    image = cv.blur(image, (10, 10))
    cv.imwrite('blur_' + filename, image)


def fourier_low_frequencies(filename):
    image = cv.imread(filename)
    rows, cols, _ = image.shape

    def filter(im):
        dft = cv.dft(np.float32(im), flags=cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
        fshift = dft_shift * mask

        back_ishift = np.fft.ifftshift(fshift)
        img_back = cv.idft(back_ishift)
        img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return cv.normalize(img_back, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    b, g, r = cv.split(image)

    r = filter(r)
    g = filter(g)
    b = filter(b)

    image = cv.merge((b, g, r))
    cv.imwrite('fourier_low_frequencies_' + filename, image)


def fourier_high_frequencies(filename):
    image = cv.imread(filename)
    rows, cols, _ = image.shape

    def filter(im):
        dft = cv.dft(np.float32(im), flags=cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        fshift = dft_shift * mask

        back_ishift = np.fft.ifftshift(fshift)
        img_back = cv.idft(back_ishift)
        img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return cv.normalize(img_back, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    b, g, r = cv.split(image)

    r = filter(r)
    g = filter(g)
    b = filter(b)

    image = cv.merge((b, g, r))
    cv.imwrite('fourier_high_frequencies_' + filename, image)


def erode(filename):
    image = cv.imread(filename)
    kernel = np.ones((5, 5), np.uint8)
    image = cv.erode(image, kernel)
    cv.imwrite('erode_' + filename, image)


def dilate(filename):
    image = cv.imread(filename)
    kernel = np.ones((5, 5), np.uint8)
    image = cv.dilate(image, kernel)
    cv.imwrite('dilate_' + filename, image)


if __name__ == '__main__':
    sources = ['dafoe.jpg']

    # Create template for colormap
    template_filename = 'template.jpg'
    r = np.array([range(256)])
    g = np.array([[0] * 256])
    b = np.array([range(255, -1, -1)])
    template = cv.merge((b, g, r))
    cv.imwrite(template_filename, template)

    for src in sources:
        orb_features(src)
        sift_features(src)
        canny_edges(src)
        grayscale(src)
        hsv(src)
        flip_vertical(src)
        flip_horizontal(src)
        rotate_45_degree(src)
        rotate_30_degree(src, 200, 300)
        shift_10_pixels(src)
        change_brightness(src, 30)
        change_contrast(src, 20)
        gamma_correction(src, 0.7)
        equalize_histogram(src)
        warming_filter(src)
        cooling_filter(src)
        colormap(src, template_filename)
        binarize(src)
        find_contours_binarized(src)
        find_contours(src)
        blur(src)
        fourier_low_frequencies(src)
        fourier_high_frequencies(src)
        erode(src)
        dilate(src)
