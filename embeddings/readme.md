Implements a Neural Probabilistic Language Model (Bengio et al., 2003).

The usage is below:

**Genral Arguments:**
- `names_file`: Path to the CSV file containing NPC names data.
- `model_path`: Path to a pre-trained model or to save a trained model.
- `window`: Context window size (number of previous tokens to consider).
- `--seed SEED`: Random seed for reproducibility. Make sure to use the same seed for train/test split if loading in a previously trained model.

**Arguments for Sampling:**
- `--sample-only`: Use this flag to generate names without training.
- `--num-names NUM_NAMES`: Number of names to generate (used with --sample-only).
- `--prefix PREFIX`: Optional prefix to start generated names (used with --sample-only).

**Arguments for Training:**
- `--batch-size BATCH_SIZE`: Batch size for training (default: 32).
- `--lr LR`: Learning rate for training (default: 0.001).
- `--epochs EPOCHS`: Number of training epochs (default: 10).
- `--hidden-size HIDDEN_SIZE`: Size of the hidden layer in the neural network.
- `--emb-size EMB_SIZE`: Size of the embeddings in the neural network.
- `--train-size TRAIN_SIZE`: Percentage of samples to use for training (default: 0.8).
- `--val-size VAL_SIZE`: Percentage of samples to use for validation (default: 0.1).
- `--eval-test`: Evaluate the final model on the test set. 

**Training Example:**
```bash
nohup python -u Embeddings.py \
  ../npc_data.csv \
  model_data/NPLM-w3-b64-lr0_005-h500-emb2 \
  3 \
  --batch-size=64 \
  --lr=0.005 \
  --epochs=120 \
  --hidden-size=500 \
  --emb-size=2 \
  --train-size=0.8 \
  --val-size=0.1 \
  --seed=42 \
  &
```

(note: -u to avoid output buffering, nohup to run in background since this may take a while)
