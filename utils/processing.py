import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

def add_start_stop_tokens(names, window):
    names = np.array(names, dtype=str)
    names = np.char.lower(names).tolist()
    names = ['<'*window + x + '>' for x in names]
    return names

def create_dataset(names, tokenizer, window):
    ix = []
    iy = []
    for word in names:
        # print(word)
        for i in range(len(word) - window):
            # print('\t', word[i:i+window], ' ---> ', word[i+window])
            ix.append(word[i:i+window])
            iy.append(word[i+window])
    
    ix_tensor = torch.tensor([tokenizer.encode(w) for w in ix])
    iy_tensor = torch.tensor(tokenizer.encode(iy))
    
    return ix_tensor, iy_tensor

def create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    # Create TensorDataset
    train_dataset, val_dataset, test_dataset = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), TensorDataset(x_test, y_test)
        
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader