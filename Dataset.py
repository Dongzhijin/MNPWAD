import torch
import time
import numpy as np
import random
from torch.nn import functional as F
import utils
import model
import time
from sklearn.cluster import KMeans
from pdb import set_trace

class DataGenerator_pretrain:
    def __init__(self, x,x_noise, y, batch_size=128,device='cuda'):
        self.x = x
        self.y = y

        self.anom_idx = np.where(self.y == 1)[0]
        self.anom_x = self.x[self.anom_idx]
        self.anom_x_noise = x_noise[self.anom_idx]
        self.norm_idx = np.where(self.y == 0)[0]
        self.norm_x = self.x[self.norm_idx]
        self.norm_x_noise = x_noise[self.norm_idx]
        self.batch_size = batch_size

    def load_batches(self, n_batches=10):
        batch_set = []
        batch_set_noise = []
        for i in range(n_batches):
            pos_idx = np.random.choice(len(self.anom_x), self.batch_size//16)
            neg_idx = np.random.choice(len(self.norm_x), self.batch_size, replace=False)
            batch = [self.anom_x[p] for p in pos_idx]+[self.norm_x[n] for n in neg_idx]
            batch_noise = [self.anom_x_noise[p] for p in pos_idx]+[self.norm_x_noise[n] for n in neg_idx]
            batch_set.append(batch)
            batch_set_noise.append(batch_noise)
        return np.array(batch_set_noise),np.array(batch_set)

class DataGenerator:
    def __init__(self, x, y, batch_size=128,device='cuda'):
        self.x = x
        self.y = y

        self.anom_idx = np.where(self.y == 1)[0]
        self.anom_x = self.x[self.anom_idx]
        self.norm_idx = np.where(self.y == 0)[0]
        self.norm_x = self.x[self.norm_idx]
        self.batch_size = batch_size
        return

    def load_batches(self, n_batches=10):
        batch_set = []
        batch_set_noise = []
        for i in range(n_batches):
            pos_idx = np.random.choice(len(self.anom_x), self.batch_size//16)
            neg_idx = np.random.choice(len(self.norm_x), self.batch_size, replace=False)
            batch = [self.anom_x[p] for p in pos_idx]+[self.norm_x[n] for n in neg_idx]
            batch_set.append(batch)
        return np.array(batch_set)
