import cv2
import pickle
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
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


def train_DCT():
	with open("models/data/cyclegan_train.npy", "rb") as f:
		x_train = np.load(f)
		y_train = np.load(f)

	print("Training model...")
	rf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
	with open("models/DCT_model.pkl", "wb") as f:
		pickle.dump(rf, f)


if __name__ == "__main__":
	calculate_train_test_data()
	train_DCT()