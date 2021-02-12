from Data.Dataset import DesignDataset

from Data.FeatureExtractor import VGG16FeatureExtractor

DATA_ROOT = "F:\\dise\\flask-dise\\static\\img\\DataSet"
VEC_ROOT = "F:\\dise\\flask-dise\\static\\feature\\VGG16\\VGG16\\DataSet"
dataset = DesignDataset(root=DATA_ROOT, vector_root=VEC_ROOT)


def test_Dataset():
    img = dataset[4:6]
    pass


def test_FeatureExtractor():
    img = dataset[4:6]
    from joblib import dump, load
    pca_ = load("./pca.pkl")
    fe = VGG16FeatureExtractor(PCA=pca_)
    res = fe(img[0][1])
    pass


if __name__ == '__main__':
    # test_Dataset()
    test_FeatureExtractor()