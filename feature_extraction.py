import numpy as np
import cv2
import matplotlib.pyplot as plt


def DCT_features(img_path):
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	imgcv = cv2.split(img)[0]
	cv2.boxFilter(imgcv, 0, (7,7), imgcv, (-1,-1), False, cv2.BORDER_DEFAULT)
	cv2.resize(imgcv, (32, 32), imgcv)

	imf = np.float32(imgcv) / 255.0
	dct = cv2.dct(imf)
	return np.uint8(dct * 255.0)


def compute_first_digits(img, name):
	dct = cv2.dct(np.float32(img) / 255.0)
	dct = np.abs(dct)

	min_val = dct.min()
	if min_val < 1:
		dct = np.power(10, -np.floor(np.log10(min_val)) + 1) * dct  # Scale all up to remove leading 0.00s

	if not (dct >= 1.0).all():
		print(img)
		print(dct)
		raise ValueError("Error")

	digits = np.log10(dct).astype(int).astype('float32')
	first_digits = dct / np.power(10, digits)
	first_digits[(first_digits < 1.0) & (first_digits > 0.9)] = 1  # Handle edge case.
	first_digits = first_digits.astype(int)

	if not (first_digits >= 1).all() and (first_digits <= 9).all():
		raise ValueError("Error")

	return first_digits


def compute_first_digits_counts(img, name):
	first_digits = compute_first_digits(img, name)
	unq, counts = np.unique(first_digits, return_counts=True)
	return unq, counts


if __name__ == "__main__":
	image = cv2.imread('data/real\\YouTube-real_00194_00030.png', cv2.IMREAD_GRAYSCALE)
	dct = cv2.dct(np.float32(image) / 255.0)
	dct = np.abs(dct)
	print(dct)
	plt.imshow(image, cmap="gray")
	plt.show()