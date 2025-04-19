import pickle

import numpy as np
import pandas as pd

data = pd.read_csv('../data/train.csv')

def create_vocab(names, window):
    names = np.array(names, dtype=str)
    names = np.char.lower(names).tolist()
    names = ['<'*window + x + '>' for x in names]

    vocab = list(set(char for word in names for char in word))
    vocab.sort()

    # Move SOS and EOS to the beginning
    vocab.remove('<')
    vocab.remove('>')
    vocab.insert(0,'>')
    vocab.insert(0, '<')

    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {v:k for k,v in stoi.items()}

    return names, vocab, stoi, itos

_, vocab, stoi, itos = create_vocab(data.name.str.lower(), 1)

vocabulary = {
    'vocab': vocab,
    'stoi': stoi,
    'itos': itos
}

with open('../data/vocabulary.pkl', 'wb') as f:
    pickle.dump(vocabulary, f)
