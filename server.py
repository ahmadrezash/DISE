import numpy as np
from PIL import Image
from datetime import datetime

from sklearn.cluster import KMeans

from flask import Flask, request, render_template

from lib import DesignDataset, DATA_ROOT
from lib.Data.FeatureExtractor import VGG16FeatureExtractor
from lib.settings import KMEANS_PATH, PCA_PATH, VEC_ROOT
import pandas as pd
from joblib import dump, load

from lib.utils import save_image

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# # ==============================
# model2= ResNet_model()
# data_loader2 = DataLoader(model_name_space="ResNet50", data_name_space="static")
# ==============================


kmeans = load(KMEANS_PATH)
pca = load(PCA_PATH)
dataset = DesignDataset(root=DATA_ROOT, vector_root=VEC_ROOT, )

fe = VGG16FeatureExtractor(PCA=pca)


# ==============================

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':

		img = dataset[10:15]
		# img_list = np.array(list(map(lambda x: x.cpu().numpy(), img[1])))
		# img_list = np.array( img[1])
		# pca = load(PCA_PATH)
		#
		# X = pca.transform(img_list)
		# n_clusters = 3
		# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

		file = request.files['query_img']

		# Save query image
		img = Image.open(file.stream)  # PIL image
		uploaded_img_path = "static/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
		img.save(uploaded_img_path)

		feature_vec = fe(img)
		lable = kmeans.predict(feature_vec.reshape(1, -1).astype(np.float32))
		sim_img_index = np.where(kmeans.labels_ == lable)
		sim_img_list = []
		scores = []
		for i in sim_img_index[0]:
			a = dataset[i]
			sim_img_list.append(dataset[i])
			scores.append(f"static/{a[2]}")

		if len(scores) > 20:
			scores = scores[:20]

		return render_template('index.html',
		                       query_path=uploaded_img_path,
		                       query_image_name=file.filename,
		                       scores=scores,
		                       model_info=[],
		                       data_loader_info=[])
	else:
		return render_template('index.html')
