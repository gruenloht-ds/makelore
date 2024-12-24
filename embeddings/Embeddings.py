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

    return ix_tensor, iy_tensor, vocab

def create_train_val_test_loader(ix_tensor, iy_tensor, train_size = 0.8, val_size=0.1, batch_size = 64):
    n1, n2 = int(len(ix_tensor) * train_size), int(len(ix_tensor) * val_size)

    x_train, x_val, x_test = ix_tensor[idx_train], ix_tensor[idx_val], ix_tensor[idx_test]
    y_train, y_val, y_test = iy_tensor[idx_train], iy_tensor[idx_val], iy_tensor[idx_test]
    
    # Create TensorDataset
    train_dataset, val_dataset, test_dataset = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), TensorDataset(x_test, y_test)
    
    # Create DataLoader
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=64), DataLoader(val_dataset, batch_size=64), DataLoader(test_dataset, batch_size=64)

    return train_loader, val_loader, test_loader

@torch.no_grad()
def eval_model(model, loader):
    val_loss = 0 
    model.eval()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        val_loss += F.cross_entropy(preds, y).item() 

    val_loss /= len(loader) 
    return val_loss


def train_model(model, train_loader, val_loader=None, epochs=10, lr=0.01, model_path=None):
    optim = torch.optim.SGD(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    val_loss = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
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
            val_loss_epoch = eval_model(model, val_loader)
            val_loss.append(val_loss_epoch)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss_epoch:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Save model
        if model_path is not None:
            torch.save(model.state_dict(), f'model_data/model-{epoch+1}.pth'))

    return (train_loss, val_loss) if val_loader else train_loss

def generate(x="", max_length=40):
    n = len(x)

    # Pad name if needed
    if n < window:
        x = '<' * (window - n) + x

    x = encode(x)
    index = len(x) - window

    # Generate name autoregressively
    while len(x) < max_length:
        tmp = model(torch.tensor(x[index:index + window])).view(-1)
        probs = F.softmax(tmp, dim=0)
        x.append(torch.multinomial(probs, 1).item())

        # Stop generation if end token is reached
        if x[-1] == encode('>')[0]:
            break

        index += 1

    # Remove start and stop tokens and return result
    return decode(torch.tensor(x)).replace("<", "").replace(">", "")

# Generate names a specified number of times, will not return anything
def generate_n(n, x=""):
    for index in range(n):
        print(generate(x))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate NPC names or train a name-generation model using embeddings - (Bengio et al., 2003)."
    )
    
    # General arguments
    parser.add_argument('window', type=int, default=3, help="Context window size (number of previous tokens to consider).")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--model-path', type=str, default=None, help="Path to a pre-trained model or to save a trained model.")
    
    # Arguments for sampling
    parser.add_argument('--sample-only', action='store_true', help="Use this flag to generate names without training.")
    parser.add_argument('--n-vocab', type=int, default=None, help="Number of characters the language model was trained on.")
    parser.add_argument('--num-names', type=int, default=1, help="Number of names to generate (used with --sample-only).")
    parser.add_argument('--prefix', type=str, default="", help="Optional prefix to start generated names (used with --sample-only).")
    
    # Arguments for training
    parser.add_argument('--names-file', type=str, default=None, help="Path to the CSV file containing NPC names data (required for training).")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training (default: 32).")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training (default: 0.001).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument('--hidden-size', type=int, default=100, help="Size of the hidden layer in the neural network.")
    parser.add_argument('--emb-size', type=int, default=2, help="Size of the embeddings in the neural network.")
    parser.add_argument('--train-size', type=float, default=0.8, help="Percentage of samples to use for training (default: 0.8).")
    parser.add_argument('--val-size', type=float, default=0.1, help="Percentage of samples to use for validation (default: 0.1).")
    parser.add_argument('--eval-test', action='store_true', help="Evaluate the final model on the test set. Make sure to use the same seed for train/test split if loading in a previously trained model.")
                            
    args = parser.parse_args()

    # Set the random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    if args.sample_only: # Generate names
        # Load model 
        model = NeuralProbabilisticLM(len(vocab), args.window, args.emb_size, args.hidden_size)
        loaded_model.load_state_dict(torch.load(args.model_path))

        generate_n(args.num_names, args.prfix)

    else: # Train a model
        # Prepare datasets
        names = pd.read_csv(args.names_file)
        ix_tensor, iy_tensor, vocab = create_dataset(names, args.window)
        train_loader, val_loader, test_loader = create_train_val_test_loader(ix_tensor, iy_tensor, args.train_size, args.val_size, args.batch_size)

        # Load model 
        model = NeuralProbabilisticLM(len(vocab), args.window, args.emb_size, args.hidden_size)
        if args.model_path is not None and os.path.exists(args.model_path):
            loaded_model.load_state_dict(torch.load(args.model_path))
        
        # Train and save model data (model, train loss, val loss)
        train_loss, val_loss = train_model(train_loader, val_loader=val_loader, epochs=args.epcohs lr=args.lr, args.model_path)
        pd.DataFrame({'train_loss': train_loss, 'val_loss': train_model}).to_csv('model_data/losses.csv', index=False)

        # Report test loss
        if args.eval_test:
            print('Test loss:'. eval_model(model, test_loader))
