import torch


def predict_grayscale():
	model = torch.load("models/grayscale.pt")
	model.eval()


if __name__ == "__main__":
	predict_grayscale()