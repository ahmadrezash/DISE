from lib.Data import DesignDataset
from sklearn.decomposition import PCA
import numpy as np
from joblib import dump, load

if __name__ == '__main__':
    DATA_ROOT = "F:\\dise\\flask-dise\\static\\img\\DataSet"
    VEC_ROOT = "F:\\dise\\flask-dise\\static\\feature\\VGG16\\VGG16\\DataSet"
    dataset = DesignDataset(root=DATA_ROOT, vector_root=VEC_ROOT)
    print("dataset created")

    img = dataset[:1000]
    print(img[1][0].shape)

    print("3000 data loaded")

    pca = PCA(n_components=600)
    l = np.array(list(map(lambda x: x.numpy(), img[1])))
    print("data reshaped")

    pca.fit(l)
    print("PCA fit")

    dump(pca, "pca.pkl")
    pca_ = load("pca.pkl")
    print("PCA saved")

    print(pca_.transform(l[:2]).shape)
