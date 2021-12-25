import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

from feature_extraction import compute_first_digits_counts


def calculate_train_test_data():
	reference_real = sorted(glob("data/cyclegan/*/*A/*.jpg"))
	reference_fake = sorted(glob("data/cyclegan/*/*B/*.jpg"))

	print("Computing Benford's Law coefficients...")
	counts_real = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE))[1] for x in reference_real])
	counts_fake = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE))[1] for x in reference_fake])
	counts_real_norm = counts_real / counts_real.sum(axis=1)[..., np.newaxis]
	counts_fake_norm = counts_fake / counts_fake.sum(axis=1)[..., np.newaxis]

	x = np.concatenate([counts_real_norm, counts_fake_norm])
	y = np.concatenate([[0 for _ in counts_real_norm], [1 for _ in counts_fake_norm]])

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0, shuffle=True)

	with open("models/data/cyclegan_train.npy", "wb") as f:
		np.save(f, x_train)
		np.save(f, y_train)
	with open("models/data/cyclegan_test.npy", "wb") as f:
		np.save(f, x_test)
		np.save(f, y_test)


def calculate_data_leave_one_group_out(groups):
	for group in groups:
		print("Computing train Benford's Law coefficients for group {}...".format(group))
		reference_real = sorted(glob("data/cyclegan/{}/*A/*.jpg".format(group)))
		reference_fake = sorted(glob("data/cyclegan/{}/*B/*.jpg".format(group)))

		counts_real = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE))[1] for x in reference_real])
		counts_fake = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE))[1] for x in reference_fake])
		counts_real_norm = counts_real / counts_real.sum(axis=1)[..., np.newaxis]
		counts_fake_norm = counts_fake / counts_fake.sum(axis=1)[..., np.newaxis]

		x = np.concatenate([counts_real_norm, counts_fake_norm])
		y = np.concatenate([[0 for _ in counts_real_norm], [1 for _ in counts_fake_norm]])

		with open("models/data/grouped/cyclegan_{}.npy".format(group), "wb") as f:
			np.save(f, x)
			np.save(f, y)
