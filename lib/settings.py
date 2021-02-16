import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATA_ROOT = "F:\\dise\\flask-dise\\static\\img\\static"
# VEC_ROOT = "F:\\dise\\flask-dise\\static\\feature\\VGG16\\VGG16\\static"

DATA_ROOT = "/home/ahmad/Project/dise/flask-dise/static/img/DataSet"
VEC_ROOT = "/home/ahmad/Project/dise/flask-dise/static/feature/VGG16/DataSet"

PCA_PATH = "./pca_vgg.pkl"
KMEANS_PATH = "./kmeans.pkl"

CLUSTER_IMAGE_PATH="./models/images_cluster"
