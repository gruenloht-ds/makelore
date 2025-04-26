import argparse
import os
import sys
import random
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

# Add directory to utils
sys.path.append(os.path.abspath('..'))

from utils.tokenizer import Tokenizer
from utils.processing import add_start_stop_tokens, create_dataset, create_dataloaders
from utils.model import eval_model, generate
from utils.general import seed_everything

# Neural Probabilistic Language Model
class NeuralProbabilisticLM(nn.Module):

    def __init__(self, n_vocab, window, emb_size, hidden_size):
        super().__init__()
        self.n_vocab = n_vocab
        self.window = window
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        
        self.C = nn.Embedding(n_vocab, emb_size)
        self.layer1 = nn.Linear(window * emb_size, hidden_size, bias=True)
        self.layer2 = nn.Linear(hidden_size, n_vocab, bias=True)

    def forward(self, x):
        x = self.C(x)
        x = x.view(-1, self.window * self.emb_size)
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        return x