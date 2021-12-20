import pickle
import numpy as np
from sklearn.metrics import confusion_matrix as cm, classification_report as cr


def predict_DCT():
	with open("models/data/cyclegan_test.npy", "rb") as f:
		x_test = np.load(f)
		y_test = np.load(f)
	with open("models/DCT_model.pkl", "rb") as f:
		rf = pickle.load(f)

	y_pred = rf.predict(x_test)
	print(cm(y_true=y_test, y_pred=y_pred))
	print(cr(y_true=y_test, y_pred=y_pred))


if __name__ == "__main__":
	predict_DCT()