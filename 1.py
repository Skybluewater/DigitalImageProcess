from matplotlib import pyplot as plt
import numpy as np
import random
import cv2
from numpy import fft


def gaussian_noise(in_sig, mu, sigma):
    in_sig_cp = np.copy(in_sig)
    m, n = in_sig_cp.shape
    for i in range(m):
        for j in range(n):
            in_sig_cp[i, j] = in_sig_cp[i, j] + random.gauss(mu, sigma)
            if in_sig_cp[i, j] < 0:
                in_sig_cp[i, j] = 0
            elif in_sig_cp[i, j] > 255:
                in_sig_cp[i, j] = 255
    return in_sig_cp


def wiener(in_sig, h, K):
    sig_cp = np.copy(in_sig)
    sig_cp_fft = fft.fft2(sig_cp)
    h_abs_square = h * np.conj(h)
    op_sig_fft = (np.conj(h) / (
            h_abs_square + K)) * sig_cp_fft  # np.conj: 返回通过改变虚部的符号而获得的共轭复数
    op_sig_shift = np.abs(np.fft.ifft2(op_sig_fft))  # 输出信号傅里叶反变换
    output = op_sig_shift
    return output


def gauss_kernel(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))


def img_process(k):
    img = cv2.imread('a.jpg', 0)
    img_blur = cv2.GaussianBlur(img, (k, k), 0)
    # 增加加性高斯白噪声
    img_gwn = gaussian_noise(img_blur, 0, 10)
    return img_blur, img_gwn, img


if __name__ == '__main__':
    k = 7
    img_add_blur, img_add_blur_and_gwn, img = img_process(k)
    img_fft = np.fft.fft2(img)
    img_add_blur_fft = np.fft.fft2(img_add_blur)
    h = img_add_blur_fft / img_fft
    img_wiener_filtering1 = wiener(img_add_blur_and_gwn, h, 0.03)
    img_wiener_filtering2 = wiener(img_add_blur_and_gwn, h, 0.3)
    img_wiener_filtering3 = wiener(img_add_blur_and_gwn, h, 0.5)
    fig = plt.figure()
    plt.subplot(2, 3, 1)
    plt.title("sourceImg")
    plt.imshow(img, cmap="gray")
    plt.subplot(2, 3, 2)
    plt.title("gaussBlur")
    plt.imshow(img_add_blur, cmap="gray")
    plt.subplot(2, 3, 3)
    plt.title("gaussBlurNoise")
    plt.imshow(img_add_blur_and_gwn, cmap="gray")
    plt.subplot(2, 3, 4)
    plt.title("wiener k = 0.03")
    plt.imshow(img_wiener_filtering1, cmap="gray")
    plt.subplot(2, 3, 5)
    plt.title("wiener k = 0.3")
    plt.imshow(img_wiener_filtering2, cmap="gray")
    plt.subplot(2, 3, 6)
    plt.title("wiener k = 0.5")
    plt.imshow(img_wiener_filtering3, cmap="gray")
    plt.show()
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("image_add_blur.jpg", img_add_blur)
    cv2.imwrite("image_add_blur_and_gwn.jpg", img_add_blur_and_gwn)
    cv2.imwrite("wiener_filtering.jpg", img_wiener_filtering1)
