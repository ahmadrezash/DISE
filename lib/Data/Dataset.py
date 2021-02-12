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

DATA_ROOT = '/home/ahmad/Project/dise/flask-dise/static/img/sample3'


class DesignDataset(data.Dataset, metaclass=ABCMeta):

    def __init__(self, root: str = DATA_ROOT, vector_root: str = "") -> None:
        # transform = InputDataTransform()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.root = root
        self.vector_root = vector_root
        self.transform = transform

    # super().__init__(root, transform=transform)

    def open_image(self, name):
        full_path = os.path.join(self.root, name)
        assert os.path.isfile(full_path)
        # img = torchvision.io.read_image(full_path).type(dtype=torch.float32)
        img = Image.open(full_path)

        return self.transform(img)

    def open_vector(self, el_name):
        f_name = el_name.split(".")[0] + ".npy"
        full_path = os.path.join(self.vector_root, f_name)
        res = np.load(full_path)
        assert os.path.isfile(full_path)

        return res

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
            vec_res = list(map(lambda x: self.open_vector(x), el_name))
        else:
            res = self.open_image(el_name)
            vec_res = self.open_vector(el_name)

        return res, vec_res

    @classmethod
    def get_images(cls, img_tensor):
        transform = ToPILImage()
        img = transform(img_tensor)
        return img


if __name__ == '__main__':
    from Dataset import DesignDataset

    # data_set = DesignDataset(root_dir="/home/ahmad/Project/dise/flask-dise/static/img/DataSet/")
    # data_set = DesignDataset(root="F:\\dise\\flask-dise\\static\\img\\DataSet")
    DATA_ROOT = "F:\\dise\\flask-dise\\static\\img\\DataSet"
    VEC_ROOT = "F:\\dise\\flask-dise\\static\\feature\\VGG16\\VGG16\\DataSet"
    dataset = DesignDataset(root=DATA_ROOT, vector_root=VEC_ROOT)
    img = dataset[4:6]
    from torchvision.models import vgg16
    model = vgg16()
    model.features(img)
