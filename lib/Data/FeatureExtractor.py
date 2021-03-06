import torch
from torchvision.models import vgg16

from lib.Data.Transforms import vgg16_transform
from lib.settings import device
import numpy as np


class VGG16FeatureExtractor(torch.nn.Module):
	def __init__(self, PCA=None):
		super(VGG16FeatureExtractor, self).__init__()
		self.transform = vgg16_transform
		model = vgg16()
		self.vgg_fe = model.features.to(device)
		self.pca = PCA

	def forward(self, obj):
		with torch.no_grad():
			try:
				preprocessed_data = self.transform(obj).unsqueeze(0).to(device)
				feature_vector = self.vgg_fe(preprocessed_data)
				flat_vector = torch.flatten(feature_vector)
				if self.pca:
					res_vector = self.pca.transform(flat_vector.unsqueeze(0).cpu()).squeeze(0)
				else:
					res_vector = np.array(flat_vector.cpu())
			except:
				res_vector = []
		return res_vector
