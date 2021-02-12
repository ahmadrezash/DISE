import torch
from torchvision.models import vgg16

from lib.Data.Transforms import vgg16_transform


class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self):
        self.transform = vgg16_transform
        model = vgg16()
        self.vgg_fe = model.features

    def forward(self, obj):
        with torch.no_grad():
            preprocessed_data = self.transform(obj).unsqueeze(0)
            feature_vector = self.vgg_fe(preprocessed_data)
            flat_vector = torch.flatten(feature_vector).squeeze(0)

        return flat_vector
