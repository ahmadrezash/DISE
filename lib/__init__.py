from torchvision.models import vgg16
from torchvision.transforms import ToTensor
from lib.Data import DesignDataset
from lib.settings import DATA_ROOT

dataset = DesignDataset(root=DATA_ROOT)

if __name__ == '__main__':
    f = ToTensor()
    a = vgg16()
    a(f(dataset[0]))