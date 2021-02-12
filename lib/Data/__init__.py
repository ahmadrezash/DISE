from torch.utils.data import DataLoader

from lib.Data.Dataset import DesignDataset

DATA_ROOT = '/home/ahmad/Project/dise/flask-dise/static/img/sample3'
dataset = DesignDataset(root=DATA_ROOT)
# Sampler = BatchSampler(sampler=)
sampler = None

data_loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=4)
