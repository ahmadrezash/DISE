import numpy as np
from lib.Data import DesignDataset
from lib.settings import DATA_ROOT, PCA_PATH, VEC_ROOT, KMEANS_PATH
from lib.utils import show_image, save_image
from joblib import dump, load
import pandas as pd
from sklearn.cluster import KMeans

if __name__ == '__main__':

	dataset = DesignDataset(root=DATA_ROOT, vector_root=VEC_ROOT, )
	img = dataset[:1000]
	img_list = np.array(list(map(lambda x: x, img[1])))

	pca = load(PCA_PATH)

	X = pca.transform(img_list)
	n_clusters = 10
	# X = list(filter(lambda x: x, X))
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

	dump(kmeans, KMEANS_PATH)

	for i in range(n_clusters):
		images = pd.DataFrame(img[0])[kmeans.labels_ == i][0]
		save_image(images=images, cols=10, name=i)
