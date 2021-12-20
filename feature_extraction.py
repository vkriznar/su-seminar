import numpy as np
import cv2


def DCT_features(img_path):
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	imgcv = cv2.split(img)[0]
	cv2.boxFilter(imgcv, 0, (7,7), imgcv, (-1,-1), False, cv2.BORDER_DEFAULT)
	cv2.resize(imgcv, (32, 32), imgcv)

	imf = np.float32(imgcv) / 255.0
	dct = cv2.dct(imf)
	return np.uint8(dct * 255.0)


def compute_first_digits(img):
	dct = cv2.dct(np.float32(img) / 255.0)
	dct = np.abs(dct)
	dct[dct == 0] = 1e-10

	min_val = dct.min()
	if min_val < 1:
		dct = np.power(10, -np.floor(np.log10(min_val)) + 1) * dct

	if not (dct >= 1.0).all():
		raise ValueError("Error")

	digits = np.log10(dct).astype(int).astype('float32')
	first_digits = dct / np.power(10, digits)
	first_digits[(first_digits < 1.0) & (first_digits > 0.9)] = 1
	first_digits = first_digits.astype(int)

	if not (first_digits >= 1).all() and (first_digits <= 9).all():
		raise ValueError("Error")

	return first_digits


def compute_first_digits_counts(img):
	first_digits = compute_first_digits(img)
	unq, counts = np.unique(first_digits, return_counts=True)
	return unq, counts
