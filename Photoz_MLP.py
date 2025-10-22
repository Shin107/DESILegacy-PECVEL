import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from astropy.table import Table, vstack , unique
import fitsio
import argparse
import time
import logging
from pathlib import Path
from Metrics import PhotoZMetrics  # Your custom metrics class
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost_cv import PhotoZCatBoostPipeline
from sklearn.preprocessing import StandardScaler

class Normalizer:
    def __init__(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MLP, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

cv_pipe = PhotoZCatBoostPipeline()
DR1, legacy = cv_pipe.load_astronomical_data()
legacy = cv_pipe.compute_magnitudes(legacy)
legacy_filtered, DR1_filtered = cv_pipe.filter_data(legacy, DR1)
X, y = cv_pipe.prepare_ml_data(legacy_filtered, DR1_filtered, feature_names=['MAG_GR', 'MAG_G', 'MAG_RZ', 'MAG_R', 'MAG_RW1', 'MAG_ZW1', 'MAG_W1W2','TYPE'])
cv_pipe.create_data_splits(X, y)




