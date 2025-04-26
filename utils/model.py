import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

@torch.no_grad()
def eval_model(model, loader, device='cpu'):
    test_loss = 0 
    model.eval()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        test_loss += F.cross_entropy(preds, y).item() 

    test_loss /= len(loader) 
    return val_loss

def generate(x="", model=None, tokenizer=None, window=None, max_length=40):
    n = len(x)

    # Pad name if needed
    if n < window:
        x = '<' * (window - n) + x

    x = tokenizer.encode(x)
    index = len(x) - window

    # Generate name autoregressively
    while len(x) < max_length:
        tmp = model.forward(torch.tensor(x[index:index + window])).view(-1)
        probs = F.softmax(tmp, dim=0)
        x.append(torch.multinomial(probs, 1).item())

        # Stop generation if end token is reached
        if x[-1] == tokenizer.end_token_id:
            break

        index += 1

    # Remove start and stop tokens and return result
    return tokenizer.decode(torch.tensor(x)).replace("<", "").replace(">", "")