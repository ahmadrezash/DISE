import os
from abc import ABCMeta

import torch
import torch.utils.data as data
import torchvision
from torchvision.transforms import ToPILImage
from torchvision import transforms
import pathlib
from PIL import Image
import numpy as np
from lib.Data.Transforms import InputDataTransform
from .FeatureExtractor import VGG16FeatureExtractor
from lib.settings import DATA_ROOT, device

fe = VGG16FeatureExtractor().to(device)


class DesignDataset(data.Dataset, metaclass=ABCMeta):

	def __init__(self, root: str = DATA_ROOT, vector_root: str = "", with_img=True) -> None:
		transform = InputDataTransform()
		# transform = t
		self.root = root
		self.vector_root = vector_root
		self.transform = transform
		self.with_img = with_img

	# super().__init__(root, transform=transform)

	def open_image(self, name):

		full_path = os.path.join(self.root, name)
		assert os.path.isfile(full_path)
		# img = torchvision.io.read_image(full_path).type(dtype=torch.float32)
		img = Image.open(full_path)

		# return self.transform(img)
		return self.transform(img)

	# def open_vector(self, img):
	#     # f_name = el_name.split(".")[0] + ".npy"
	#     # full_path = os.path.join(self.vector_root, f_name)
	#     # res = np.load(full_path)
	#     # assert os.path.isfile(full_path)
	#     # full_path = os.path.join(self.root, name)
	#     # img = Image.open(full_path)
	#     return self.transform(img)

	def get_abs_path(self):
		return pathlib.Path(self.root)

	@property
	def data_list(self):
		abs_path = self.get_abs_path()
		res = os.listdir(abs_path)
		return res

	@property
	def vector_list(self):
		abs_path = pathlib.Path(self.vector_root)
		res = os.listdir(abs_path)
		return res

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		el_name = self.data_list[idx]

		if type(el_name) is list:
			res = list(map(lambda x: self.open_image(x), el_name))
			vec_res = list(map(lambda x: fe(x), res))
		else:
			res = self.open_image(el_name)
			vec_res = self.transform(res)
		if self.with_img:
			return res, vec_res, el_name
		else:
			return [], vec_res, el_name

	@classmethod
	def get_images(cls, img_tensor):
		transform = ToPILImage()
		img = transform(img_tensor)
		return img
