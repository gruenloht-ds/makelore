import pandas as pd
import numpy as np

from collections import defaultdict
import argparse
import random

class MarkovModel():

    def __init__(self, smoothing_value, window):
        
        self.smoothing_value = smoothing_value
        self.window = window
        
        # Initialize a nested defaultdict
        self.counter = defaultdict(lambda: defaultdict(int))

    def fit(self, x):

        # Start and stop tokens will be "<" and '>' respectively
        self.corpus = set("".join(x))
        self.corpus.add('>')
        
        # Add start and end padding to each name
        x = np.array(['<'*(self.window) + name + '>'*(self.window) for name in x])
        
        # Build the counter with observed transitions
        for name in x:
            for start in range(len(name) - self.window):
                end = start + self.window
                n_gram = name[start:end]
                next_word = name[end]
                self.counter[n_gram][next_word] += 1

        # If smoothing is applied add regularization to the counts
        if self.smoothing_value > 0:
            for n_gram in self.counter:
                for char in self.corpus:
                    self.counter[n_gram][char] += self.smoothing_value

        # Converts to probabilities
        self.normalize()

        # sort inner dicts by keys
        self.counter = {k: dict(sorted(v.items())) for k, v in self.counter.items()} # bro why do i need this
        # For some reason - only when I run the code with a set seed using the terminal (using jupyter notebook is fine and works as expected)
        # the names generate slightly differently even with a set seed (again only happens when using the terminal to run this - dont ask me why)
        # sorting the dict before generating ensures the same name is generated each time when using a seed (yes, this took hours of troubleshooting)

    def normalize(self):
        # Normalize counts to probabilities
        for n_gram, next_words in self.counter.items():
            total = sum(next_words.values())
            for next_word in next_words:
                self.counter[n_gram][next_word] /= total

    # Functions to generate names
    def generate(self, x = ""):
        n = len(x)
        # Pad name if not already needed
        if n < self.window:
            x = '<'*(self.window - n) + x

        # Generate name autoregressively
        while True:
            last_n_gram = x[-self.window:]
            next_char_prob = self.counter.get(last_n_gram, {})
            
            # If no possible next character, break the loop
            if not next_char_prob:
                break
            
            next_char = np.random.choice(list(next_char_prob.keys()), p=list(next_char_prob.values()))
            x += next_char
            
            # Stop if the end token is reached
            if next_char == '>':
                break

        # Remove start and stop tokens
        return x.replace("<", "").replace(">", "")

    # Generate names a specified number of times, will not return anything - just prints
    def generate_n(self, n, x=""):
        for index in range(n):
            print(self.generate(x))


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate names using a Markov model.")
    parser.add_argument('names_file', type=str, help="The provided NPC names data csv file")
    parser.add_argument('window', type=int, help="The n-gram window size.")
    parser.add_argument('smoothing_value', type=float, help="Smoothing value to apply.")
    parser.add_argument('--num_names', type=int, default=1, help="Number of names to generate.")
    parser.add_argument('--prefix', type=str, default="", help="Optional prefix to start generated names.")
    parser.add_argument('--seed', type=int, default=None, help="Seed to generate the same names each time.")
                        
    args = parser.parse_args()

    # Set the random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    names = pd.read_csv(args.names_file).name.values
    names = np.array(names, dtype=str)
    names = np.char.lower(names)

    # Initialize the model
    model = MarkovModel(smoothing_value=args.smoothing_value, window=args.window)
    model.fit(names)
    
    # Generate the specified number of names
    model.generate_n(args.num_names, x=args.prefix)
