import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

import argparse
import os
import random
import pickle

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

def create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    # Create TensorDataset
    train_dataset, val_dataset, test_dataset = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), TensorDataset(x_test, y_test)
        
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

@torch.no_grad()
def eval_model(loader, device='cpu'):
    val_loss = 0 
    model.eval()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        val_loss += F.cross_entropy(preds, y).item() 

    val_loss /= len(loader) 
    return val_loss


def train_model(train_loader, val_loader, epochs=10, lr=0.01, model_path=None, device='cpu'):
    optim = torch.optim.SGD(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        loss_accum = 0.0
        model.train()

        # Training loop
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)

            # Compute loss and perform backpropagation
            loss = criterion(preds, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_accum += loss.item()

        # Track training loss
        avg_train_loss = loss_accum / len(train_loader)
        train_loss.append(avg_train_loss)

        # Evaluate on validation set if provided
        if val_loader:
            model.eval()
            val_loss_epoch = eval_model(val_loader)
            val_loss.append(val_loss_epoch)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss_epoch:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Save model
        if model_path is not None:
            torch.save(model.state_dict(), f'{model_path}-{epoch+1}.pth')

    return (train_loss, val_loss) if val_loader else train_loss

def generate(x="", window=None, stoi=None, itos=None, max_length=40):
    n = len(x)

    # Pad name if needed
    if n < window:
        x = '<' * (window - n) + x

    x = encode(x, stoi)
    index = len(x) - window

    # Generate name autoregressively
    while len(x) < max_length:
        tmp = model(torch.tensor(x[index:index + window])).view(-1)
        probs = F.softmax(tmp, dim=0)
        x.append(torch.multinomial(probs, 1).item())

        # Stop generation if end token is reached
        if x[-1] == encode('>', stoi)[0]:
            break

        index += 1

    # Remove start and stop tokens and return result
    return decode(torch.tensor(x), itos).replace("<", "").replace(">", "")

# Generate names a specified number of times, will not return anything
def generate_n(n, window, stoi=None, itos=None, x="", max_len=40):
    for index in range(n):
        print(generate(x, window, stoi, itos, max_len))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate NPC names or train a name-generation model using embeddings - (Bengio et al., 2003)."
    )
    
    # General arguments
    parser.add_argument('vocab_path', type=str, default=None, help="Path to the vocabulary (.pkl file).")
    parser.add_argument('save_model_path', type=str, default=None, help="Path to save the trained model (without the extension).")
    parser.add_argument('window', type=int, default=3, help="Context window size (number of previous tokens to consider).")
    parser.add_argument('hidden_size', type=int, default=100, help="Size of the hidden layer in the neural network.")
    parser.add_argument('emb_size', type=int, default=2, help="Size of the character embeddings in the neural network.")
    parser.add_argument('--data-path', type=str, default=None, help="Path to the processed datasets (.pkl file).")
    parser.add_argument('--load-model-path', type=str, default=None, help="Path to the pre-trained model (with the extension).")
    parser.add_argument('--seed', type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--n_threads", type=int, default=torch.get_num_threads(),help="Number of CPU threads to use.")
    
    # Arguments for sampling
    parser.add_argument('--sample-only', action='store_true', help="Use this flag to generate names without training.")
    parser.add_argument('--num-names', type=int, default=1, help="Number of names to generate (used with --sample-only).")
    parser.add_argument('--prefix', type=str, default="", help="Optional prefix to start generated names (used with --sample-only).")
    
    # Arguments for training
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training (default: 32).")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training (default: 0.001).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument('--eval-test', action='store_true', help="Evaluate the final model on the test set.")
                            
    args = parser.parse_args()

    # Set the number of threads PyTorch uses
    torch.set_num_threads(args.n_threads)
    torch.set_num_interop_threads(args.n_threads)

    # Set the random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    # Load vocabulary
    with open(args.vocab_path, "rb") as file:
        vocabulary = pickle.load(file)

    vocab, stoi, itos = vocabulary.values()
    
    if args.sample_only: # Generate names
        # Load model
        model = NeuralProbabilisticLM(len(vocab), args.window, args.emb_size, args.hidden_size)
        model.load_state_dict(torch.load(args.load_model_path, weights_only=True))

        generate_n(args.num_names, args.window, stoi, itos, args.prefix)

    else: # Train a model
        # Load datasets
        with open(args.data_path, "rb") as file:
            data = pickle.load(file)

        data['batch_size'] = args.batch_size
        
        # Prepare datasets
        train_loader, val_loader, test_loader = create_dataloaders(**data)
        
        # Load model 
        model = NeuralProbabilisticLM(len(vocab), args.window, args.emb_size, args.hidden_size)
        if args.load_model_path is not None and os.path.exists(args.load_model_path):
            model.load_state_dict(torch.load(args.load_model_path, weights_only=True))

        # Train and save model data (model, train loss, val loss)
        train_loss, val_loss = train_model(train_loader, val_loader=val_loader, epochs=args.epochs, lr=args.lr, model_path=args.save_model_path)
        pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss}).to_csv('model_data/losses.csv', index=False)

        # Report test loss
        if args.eval_test:
            print('Test loss:', eval_model(test_loader))