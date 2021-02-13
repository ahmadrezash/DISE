from lib.Data import DesignDataset
from sklearn.decomposition import PCA
import numpy as np
from joblib import dump, load
from lib.settings import DATA_ROOT, VEC_ROOT, PCA_PATH

if __name__ == '__main__':
	dataset = DesignDataset(root=DATA_ROOT, vector_root=VEC_ROOT)
	print("dataset created")

	count_of_data = 1500
	output_dim = 600

	img = dataset[:count_of_data]
	print(img[1][0].shape)

	print("3000 data loaded")

	pca = PCA(n_components=output_dim)
	img_list = np.array(list(map(lambda x: x.cpu().numpy(), img[1])))
	print("data reshaped")

	pca.fit(img_list)
	print("PCA fit")

	dump(pca, PCA_PATH)
	pca_ = load(PCA_PATH)
	print("PCA saved")

	print("Test loaded PCA")
	print(pca_.transform(img_list[:2]).shape)
