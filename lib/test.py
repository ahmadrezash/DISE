from Data.Dataset import DesignDataset
from Data.FeatureExtractor import VGG16FeatureExtractor

from lib.settings import VEC_ROOT, DATA_ROOT,PCA_PATH

dataset = DesignDataset(root=DATA_ROOT, vector_root=VEC_ROOT)


def test_Dataset():
	img = dataset[4:6]
	pass


def test_FeatureExtractor():
	img = dataset[4:6]
	from joblib import dump, load
	pca_ = load(PCA_PATH)
	fe = VGG16FeatureExtractor(PCA=pca_)
	res = fe(img[0][1])
	pass

def test_PerceptualLoss():
	from lib.Data.PerceptualLoss import VGGPerceptualLoss
	from torchvision.transforms import ToTensor
	img = dataset[4:6]
	dis_model = VGGPerceptualLoss()
	t = ToTensor()
	dis = dis_model(img[0][0],img[0][1])

	print(dis)


if __name__ == '__main__':
	# test_Dataset()
	# test_FeatureExtractor()

	test_PerceptualLoss()