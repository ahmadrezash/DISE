import torch
from torchvision.models import vgg16

from lib.Data.Transforms import vgg16_transform


class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self, PCA=None):
        super(VGG16FeatureExtractor, self).__init__()
        self.transform = vgg16_transform
        model = vgg16()
        self.vgg_fe = model.features
        self.pca = PCA

    def forward(self, obj):
        with torch.no_grad():
            preprocessed_data = self.transform(obj).unsqueeze(0)
            feature_vector = self.vgg_fe(preprocessed_data)
            flat_vector = torch.flatten(feature_vector)
            if self.pca:
                res_vector = self.pca.transform(flat_vector.unsqueeze(0)).squeeze(0)
            else:
                res_vector = flat_vector

        return res_vector
