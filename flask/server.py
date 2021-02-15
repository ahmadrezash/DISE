import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template

from lib import DesignDataset, DATA_ROOT
from lib.Data.FeatureExtractor import VGG16FeatureExtractor
from lib.settings import KMEANS_PATH, PCA_PATH, VEC_ROOT
import pandas as pd
from joblib import dump, load

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# # ==============================
# model2= ResNet_model()
# data_loader2 = DataLoader(model_name_space="ResNet50", data_name_space="DataSet")
# ==============================


kmeans = load(KMEANS_PATH)
pca = load(PCA_PATH)
dataset = DesignDataset(root=DATA_ROOT, vector_root=VEC_ROOT, )

fe = VGG16FeatureExtractor(PCA=pca)


# ==============================

@app.route('/<model>', methods=['GET', 'POST'])
def index(model):
    if request.method == 'POST':

        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        feature_vec = fe(img)
        lable = kmeans.predict(feature_vec)
        pd.DataFrame(img[0])[kmeans.labels_ == lable]
        dataset

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               query_image_name=file.filename,
                               scores=[],
                               model_info=[],
                               data_loader_info=[])
    else:
        return render_template('index.html')
