import cv2
import numpy as np
from src.utils import show_image, add_noise_sp, add_noise_g
from src.tgv import alg3, tgv_denoise_pd


def to_float(image_raw):
    return image_raw.astype(np.float32) / 255 - 0.5


def to_uint8(image_f):
    return ((image_f + 0.5) * 255).astype(np.uint8)


image_raw = cv2.imread("data/peppers.png", cv2.IMREAD_GRAYSCALE)
image_f = to_float(image_raw)
image_f = add_noise_g(image_f, std=0.3)

image_denoised_alg3 = alg3(image_f, lambda_tv=16.0, n_it=100, alpha=0.05, L=12)
image_denoised_pd = tgv_denoise_pd(image_f, lambda_tv=0.03, n_it=1000, alpha=0.05, L=12)

cv2.imwrite("output/image.png", to_uint8(image_f))
cv2.imwrite("output/image_denoised_alg3.png", to_uint8(image_denoised_alg3))
cv2.imwrite("output/image_denoised_pd.png", to_uint8(image_denoised_pd))
cv2.waitKey(10)