import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

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

def encode(word):
    return [stoi[c] for c in word]

def decode(ix):
    return ''.join([itos[i] for i in ix.tolist()])

def create_dataset(names, window):
    names = np.array(names, dtype=str)
    names = np.char.lower(names).tolist()
    names = ['<'*window + x + '>' for x in names]

    vocab = list(set(char for word in names for char in word))

    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {v:k for k,v in stoi.items()}

    ix = []
    iy = []
    for word in names:
        # print(word)
        for i in range(len(word) - window):
            # print('\t', word[i:i+window], ' ---> ', word[i+window])
            ix.append(word[i:i+window])
            iy.append(word[i+window])

    ix_tensor = torch.tensor([encode(w) for w in ix])
    iy_tensor = torch.tensor(encode(iy))

    return ix_tensor, iy_tensor

def create_train_val_test_loader(ix_tensor, iy_tensor, train_size = 0.8, val_size=0.1, batch_size = 64, seed = None):
    n1, n2 = int(len(ix_tensor) * train_size), int(len(ix_tensor) * val_size)

    x_train, x_val, x_test = ix_tensor[idx_train], ix_tensor[idx_val], ix_tensor[idx_test]
    y_train, y_val, y_test = iy_tensor[idx_train], iy_tensor[idx_val], iy_tensor[idx_test]
    
    # Create TensorDataset
    train_dataset, val_dataset, test_dataset = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), TensorDataset(x_test, y_test)
    
    # Create DataLoader
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=64), DataLoader(val_dataset, batch_size=64), DataLoader(test_dataset, batch_size=64)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate NPC names or train a name-generation model using embeddings - (Bengio et al., 2003)."
    )
    
    # General arguments
    parser.add_argument('window', type=int, help="Context window size (number of previous tokens to consider).")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--model-path', type=str, default=None, help="Path to a pre-trained model or to save a trained model.")
    
    # Arguments for sampling
    parser.add_argument('--sample-only', action='store_true', help="Use this flag to generate names without training.")
    parser.add_argument('--num-names', type=int, default=1, help="Number of names to generate (used with --sample-only).")
    parser.add_argument('--prefix', type=str, default="", help="Optional prefix to start generated names (used with --sample-only).")
    
    # Arguments for training
    parser.add_argument('--names-file', type=str, default=None, help="Path to the CSV file containing NPC names data (required for training).")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training (default: 32).")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training (default: 0.001).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument('--hidden-size', type=int, default=100, help="Size of the hidden layer in the neural network.")
                            
    args = parser.parse_args()

    # Set the random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    #### TODO



