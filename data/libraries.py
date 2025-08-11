# libraries.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms as transforms

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.impute import SimpleImputer, KNNImputer
import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from scipy import stats
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import statsmodels.api as sm

import json
import pickle
import joblib
from pathlib import Path
import os
import glob
import requests
import zipfile

try:
    import xarray as xr
    import netCDF4
except ImportError:
    pass

try:
    import wbdata
except ImportError:
    pass

from datetime import datetime, timedelta
import calendar
from collections import Counter, defaultdict
from itertools import combinations
import random
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)
random.seed(42)

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices('GPU')[0], True
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")