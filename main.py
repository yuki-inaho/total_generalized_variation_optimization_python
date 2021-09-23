import cv2
import numpy as np
from src.utils import show_image, add_noise, add_noise_g
from src.tgv import alg3, tgv_denoise_pd

image_raw = cv2.imread("data/lena2.png", cv2.IMREAD_GRAYSCALE)
h, w = image_raw.shape
clip = np.minimum(h, w)
image_raw = image_raw[(h // 2 - clip // 2) : (h // 2 + clip //2), (w // 2 - clip //2) : (w // 2 + clip //2)]

# image_raw = cv2.imread("data/lena.jpg", cv2.IMREAD_GRAYSCALE)

# image_raw = cv2.rotate(image_raw, cv2.ROTATE_90_CLOCKWISE)
# image_raw = add_noise_g(image_raw, var=0.5).astype(np.float32)

# grad = gradient(image.copy())  # [0, :, :] -> y-diff image, [1, :, :] -> x-diff image
# div = gradience(grad)

image_f = image_raw.astype(np.float32) / 255 - 0.5
# image_f = image_raw.astype(np.float32) / 255
image_f = add_noise_g(image_f, var=0.005).astype(np.float32)

# image_new = tvl1(image_f, lambda_tv=1.0, n_it=1000)
# odl_pdhg(image_f)
# image_new = alg3(image_f, lambda_tv=32.0, n_it=500, alpha=0.05, L=12)
image_new = tgv_denoise_pd(image_f, lambda_tv=0.05, n_it=500, L=24.0)
# image_new = alg2(image_f, lambda_tv=16.0, n_it=500)

cv2.imwrite("output/image.png", ((image_f + 0.5) * 255).astype(np.uint8))
cv2.imwrite("output/image_denoised.png", ((image_new + 0.5) * 255).astype(np.uint8))
cv2.waitKey(10)