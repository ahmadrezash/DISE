from torchvision.transforms import Compose, Grayscale, Resize, ToPILImage, transforms,ToTensor


class InputDataTransform(Compose):

    def __init__(self):
        transforms = [
            Resize((100, 100)),
            # ToTensor()
            # Grayscale()
        ]
        super().__init__(transforms)


class OutputDataTransform(Compose):

    def __init__(self):
        transforms = [
            ToPILImage()
        ]
        super().__init__(transforms)


vgg16_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
