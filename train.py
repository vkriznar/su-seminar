import pickle
import itertools
import numpy as np
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report as cr


def train_DCT():
	with open("models/data/cyclegan_train.npy", "rb") as f:
		x_train = np.load(f)
		y_train = np.load(f)

	print("Training model...")
	rf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
	with open("models/DCT_model.pkl", "wb") as f:
		pickle.dump(rf, f)


def train_majority():
	with open("models/data/cyclegan_train.npy", "rb") as f:
		x_train = np.load(f)
		y_train = np.load(f)

	print("Training model...")
	rf = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
	with open("models/majority_model.pkl", "wb") as f:
		pickle.dump(rf, f)


def preload_group(group):
	with open("models/data/grouped/cyclegan_{}.npy".format(group), "rb") as f:
		x_train = np.load(f)
		y_train = np.load(f)
	return (x_train, y_train)


def train_predict_group(groups):
	preloaded_data = [preload_group(group) for group in groups]
	for i, group in enumerate(groups):
		print("Training {} model...".format(group))
		group_excluded_data = preloaded_data[:i] + preloaded_data[i+1:]
		x_train, y_train = zip(*group_excluded_data)
		x_train, y_train = shuffle(list(itertools.chain.from_iterable(x_train)), list(itertools.chain.from_iterable(y_train)), random_state=0)
		x_test, y_test = preloaded_data[i]
		rf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
		y_pred = rf.predict(x_test)
		print("DCT results for group {}".format(group))
		print(cr(y_true=y_test, y_pred=y_pred))

		rf = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
		y_pred = rf.predict(x_test)
		print("Majority results for group {}".format(group))
		print(cr(y_true=y_test, y_pred=y_pred, zero_division=0))
		print("\n")
