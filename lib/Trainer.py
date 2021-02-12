import torch
from torchvision.utils import save_image
from .utils import to_img


class TrainerBase:
	def __init__(self, data_loader, model_class, epochs):

		# Dataset
		self.data_loader = data_loader

		# Model Config
		self.model = model_class()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = self.model.to(self.device)

		# HyperParams
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
		self.criterion = torch.nn.MSELoss()
		self.epochs = epochs

	def evaluate_model(test_dl, model):
		pass

	def watchman(self, outputs, epoch, loss):
		pic = to_img(outputs.cpu().data)
		save_image(pic, './data/img_res/image_{}.png'.format(epoch))
		print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, self.epochs, loss))

	def tensorboard_logger(self, epoch, loss):
		pass

	def train(self):

		optimizer = self.optimizer
		criterion = self.criterion

		epochs = self.epochs

		print('=============== Starting Training ===============')
		for epoch in range(epochs):
			loss = 0
			for i, batch_features in enumerate(self.data_loader, 0):
				batch_features = batch_features.to(self.device)

				# compute loss and update
				optimizer.zero_grad()
				outputs = self.model(batch_features)
				train_loss = criterion(outputs, batch_features)
				train_loss.backward()
				optimizer.step()

				loss += train_loss.item()

			loss = loss / len(self.data_loader)

			if epoch % 5 == 0:
				self.watchman(outputs, epoch, loss)
		print('=============== Finished Training ===============')
