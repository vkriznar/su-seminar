import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from feature_extraction import compute_first_digits_counts


def preload_group(group):
	with open("models/data/grouped/cyclegan_{}.npy".format(group), "rb") as f:
		x_train = np.load(f)
		y_train = np.load(f)
	return (x_train, y_train)


def analyse_benford():
	reference_real = sorted(glob("data/cyclegan/facades/*A/*.jpg"))
	reference_fake = sorted(glob("data/cyclegan/facades/*B/*.jpg"))
	counts_real_orig = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE))[1] for x in reference_real])
	counts_fake_orig = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE))[1] for x in reference_fake])
	counts_real_norm = counts_real_orig / counts_real_orig.sum(axis=1)[..., np.newaxis]
	counts_fake_norm = counts_fake_orig / counts_fake_orig.sum(axis=1)[..., np.newaxis]

	counts_real_zipped = zip(*counts_real_norm)
	counts_fake_zipped = zip(*counts_fake_norm)

	counts_real = [sum(x)/len(x) for x in counts_real_zipped]
	counts_fake = [sum(x)/len(x) for x in counts_fake_zipped]

	x = np.arange(1, 10)
	benford = np.log10(1 + 1 / x)
	plt.bar(x, benford, color='c', label="Benford's law")
	plt.plot(x, counts_real, color='orange', marker='x', linestyle='dashed', linewidth=2, markersize=12, label="Real images")
	plt.plot(x, counts_fake, color='purple', marker='x', linestyle='dashed', linewidth=2, markersize=12, label="GAN generated")
	plt.xlabel("Most significant digit")
	plt.ylabel("p(d)")
	plt.legend()
	plt.show()


def show_motion_blur():
	image = cv2.imread("data/cyclegan/horse2zebra/testA/n02381460_840.jpg")
	plt.imshow(image[:, :, ::-1])
	plt.show()

	image_blurred = cv2.medianBlur(image, 3)
	plt.imshow(image_blurred[:, :, ::-1])
	plt.show()
