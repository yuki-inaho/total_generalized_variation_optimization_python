import cv2
import numpy as np


def show_image(image, title="image"):
    while cv2.waitKey(10) & 0xFF not in [ord("q"), 27]:
        cv2.imshow(title, image)
    cv2.destroyAllWindows()


def normalize(image):
    return ((image + 0.5) * 255).astype(np.uint8)


def save_image(image, title="image.jpg", enable_normalize=True):
    if enable_normalize:
        image = normalize(image)
    cv2.imwrite(title, image)
    cv2.waitKey(10)


def add_noise(image, s_vs_p=0.5, amount=0.004):
    row, col = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out


def add_noise_g(image, var=0.1):
    row, col = image.shape
    mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy