import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATA_ROOT = "F:\\dise\\flask-dise\\static\\img\\DataSet"
# VEC_ROOT = "F:\\dise\\flask-dise\\static\\feature\\VGG16\\VGG16\\DataSet"

DATA_ROOT = "/home/ahmad/Project/dise/flask-dise/static/img/DataSet"
VEC_ROOT = "/home/ahmad/Project/dise/flask-dise/static/feature/VGG16/DataSet"

PCA_PATH = "./pca_vgg.pkl"
