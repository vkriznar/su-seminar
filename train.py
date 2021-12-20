import torch
import numpy as np
from glob import glob
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm, classification_report as cr
from classifier.random_forest_classifier import TorchRandomForestClassifier
from feature_extraction import compute_first_digits_counts


def train_grayscale():
	model = TorchRandomForestClassifier(nb_trees=100, nb_samples=3, max_depth=5, bootstrap=True)
	data = torch.FloatTensor([[0,1,2.4],[1,2,1],[4,2,0.2],[8,3,0.4], [4,1,0.4]])
	labels = torch.LongTensor([0,0,0,0,0])

	model.fit(data, labels)
	torch.save(model, "models/grayscale.pt")


def train_DCT():
	reference_real = sorted(glob("data/real/*.png"))
	reference_fake = sorted(glob("data/fake/*.png"))

	print("Computing Benford's Law coefficients...")
	counts_real = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE), x)[1] for x in reference_real])[:100]
	counts_fake = np.array([compute_first_digits_counts(cv2.imread(x, cv2.IMREAD_GRAYSCALE), x)[1] for x in reference_fake])[:100]
	counts_real_norm = counts_real / counts_real.sum(axis=1)[..., np.newaxis]
	counts_fake_norm = counts_fake / counts_fake.sum(axis=1)[..., np.newaxis]

	x = np.concatenate([counts_real_norm, counts_fake_norm])
	y = np.concatenate([[0 for _ in counts_real_norm], [1 for _ in counts_fake_norm]])

	print("Training model...")
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0, shuffle=True)
	rf = RandomForestClassifier(random_state=0).fit(x_train, y_train)

	y_pred = rf.predict(x_train)
	print(cm(y_true=y_train, y_pred=y_pred))
	print(cr(y_true=y_train, y_pred=y_pred))

	y_pred = rf.predict(x_test)
	print(cm(y_true=y_test, y_pred=y_pred))
	print(cr(y_true=y_test, y_pred=y_pred))


if __name__ == "__main__":
	train_DCT()