import pickle
import numpy as np
import cv2
from glob import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report as cr
from sklearn.model_selection import train_test_split

from feature_extraction import compute_first_digits_counts


def predict_DCT():
	with open("models/data/cyclegan_test.npy", "rb") as f:
		x_test = np.load(f)
		y_test = np.load(f)
	with open("models/DCT_model.pkl", "rb") as f:
		rf = pickle.load(f)

	y_pred = rf.predict(x_test)
	print(cr(y_true=y_test, y_pred=y_pred))


def predict_majority():
	with open("models/data/cyclegan_test.npy", "rb") as f:
		x_test = np.load(f)
		y_test = np.load(f)
	with open("models/majority_model.pkl", "rb") as f:
		rf = pickle.load(f)

	y_pred = rf.predict(x_test)
	print(cr(y_true=y_test, y_pred=y_pred, zero_division=0))


def train_predict_motion_blur():
	reference = sorted(glob("data/cyclegan/horse2zebra/*A/*.jpg"))

	counts_real = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE))[1] for x in reference])
	counts_fake = np.array([compute_first_digits_counts(cv2.medianBlur(cv2.imread(x, cv2.IMREAD_GRAYSCALE), 3))[1] for x in reference])
	counts_real_norm = counts_real / counts_real.sum(axis=1)[..., np.newaxis]
	counts_fake_norm = counts_fake / counts_fake.sum(axis=1)[..., np.newaxis]

	x = np.concatenate([counts_real_norm, counts_fake_norm])
	y = np.concatenate([[0 for _ in counts_real_norm], [1 for _ in counts_fake_norm]])

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=True)

	rf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
	y_pred = rf.predict(x_test)
	print(cr(y_true=y_test, y_pred=y_pred))

	rf = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
	y_pred = rf.predict(x_test)
	print(cr(y_true=y_test, y_pred=y_pred))
