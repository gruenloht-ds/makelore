import pandas as pd
import numpy as np

import torch

import argparse
import os
import random
import pickle

def encode(word, stoi):
    return [stoi[c] for c in word]

def decode(ix, itos):
    return ''.join([itos[i] for i in ix.tolist()])
    
def create_vocab(names, window):
    names = np.array(names, dtype=str)
    names = np.char.lower(names).tolist()
    names = ['<'*window + x + '>' for x in names]
    
    vocab = list(set(char for word in names for char in word))
    vocab.sort() # Sort to keep reproducability across runs
    
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {v:k for k,v in stoi.items()}
    
    return names, vocab, stoi, itos
    
def create_dataset(names, window, stoi, itos):
    ix = []
    iy = []
    for word in names:
        # print(word)
        for i in range(len(word) - window):
            # print('\t', word[i:i+window], ' ---> ', word[i+window])
            ix.append(word[i:i+window])
            iy.append(word[i+window])
    
    ix_tensor = torch.tensor([encode(w, stoi) for w in ix])
    iy_tensor = torch.tensor(encode(iy, stoi))
    
    return ix_tensor, iy_tensor
    
def create_train_val_test_loader(ix_tensor, iy_tensor, train_size = 0.8, val_size=0.1):
    val_size = train_size + val_size
    n1, n2 = int(len(ix_tensor) * train_size), int(len(ix_tensor) * val_size)

    idx = torch.randperm(len(ix_tensor))
    idx_train, idx_val, idx_test = idx[:n1], idx[n1:n2], idx[n2:]
    
    x_train, x_val, x_test = ix_tensor[idx_train], ix_tensor[idx_val], ix_tensor[idx_test]
    y_train, y_val, y_test = iy_tensor[idx_train], iy_tensor[idx_val], iy_tensor[idx_test]
        
    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process names data into train, val, and test sets"
    )
    
    parser.add_argument('names_file', type=str, default=None, help="Path to the CSV file containing NPC names data.")
    parser.add_argument('save_data_path', type=str, default=None, help="Path to save the processed data (must be .pkl file).")
    parser.add_argument('save_vocab_path', type=str, default=None, help="Path to save the vocabulary (must be .pkl file).")
    parser.add_argument('window', type=int, default=3, help="Context window size (number of previous tokens to consider).")
    parser.add_argument('--seed', type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument('--train-size', type=float, default=0.8, help="Percentage of samples to use for training (default: 0.8).")
    parser.add_argument('--val-size', type=float, default=0.1, help="Percentage of samples to use for validation (default: 0.1).")
                            
    args = parser.parse_args()

    # Set the random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.manual_seed(args.seed)

    # Process names
    names = pd.read_csv(args.names_file).name.values
    names, vocab, stoi, itos = create_vocab(names, args.window)
        
    # Prepare datasets
    ix_tensor, iy_tensor = create_dataset(names, args.window, stoi, itos)
    x_train, y_train, x_val, y_val, x_test, y_test = create_train_val_test_loader(ix_tensor, iy_tensor, args.train_size, args.val_size)

    # Store data as dictionary
    processed_data = {
        'x_train': x_train, 
        'y_train': y_train, 
        'x_val': x_val, 
        'y_val': y_val, 
        'x_test': x_test, 
        'y_test': y_test
    }

    vocabulary = {
        'vocab': vocab,
        'stoi': stoi,
        'itos': itos
    }

    # Save data as pickle file
    with open(args.save_data_path, "wb") as file:
        pickle.dump(processed_data, file)

    if args.save_vocab_path is not None:
        with open(args.save_vocab_path, "wb") as file:
            pickle.dump(vocabulary, file)
