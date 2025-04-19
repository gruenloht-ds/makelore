import argparse
import pickle
import numpy as np
import pandas as pd
import torch

class Tokenizer():

    def __init__(self, start_token = '<', end_token = '>', unk_token = '^'):
        self.vocab = []
        self.stoi = dict()
        self.itos = dict()
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token

    def create_vocab(self, names):
        """
        Creates vocab of names
        Creates index to string and string to index
        """
        names = np.array(names, dtype=str)
        names = np.char.lower(names).tolist()

        self.vocab = list(set(char for word in names for char in word if not (char == '<' or char == '>')))
        self.vocab.sort()

        # Add SOS, EOS, and UNK to the beginning - the vocab does not already contain these characters
        self.vocab.insert(0, self.unk_token)
        self.vocab.insert(0, self.end_token)
        self.vocab.insert(0, self.start_token)

        self.unk_token_id = 2
        self.end_token_id = 1
        self.start_token_id = 0

        self.stoi = {w:i for i,w in enumerate(self.vocab)}
        self.itos = {v:k for k,v in self.stoi.items()}

        self.vocabulary = [self.vocab, self.stoi, self.itos]

    def fit(self, names):
        self.create_vocab(names)

    def encode(self, word):
        return [self.stoi[c] if c in self.vocab else self.unk_token_id for c in word]

    def decode(self, ix):
        if type(ix) == torch.tensor or type(ix) == torch.Tensor:
            ix = ix.tolist()
        return ''.join([self.itos[i] for i in ix])


def main():
    parser = argparse.ArgumentParser(description="Train and save a Tokenizer.")
    parser.add_argument('--names-file', type=str, required=True, help='Optional path to a file with names, one per line.')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save the tokenizer pickle file.')

    args = parser.parse_args()

    data = pd.read_csv(args.names_file)
    names = data.name

    tokenizer = Tokenizer()
    tokenizer.fit(names)

    with open(args.save_path, 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == '__main__':
    main()
