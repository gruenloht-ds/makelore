Implements a Neural Probabilistic Language Model (Bengio et al., 2003).

To generate names after training the model:
```bash
python Embeddings.py model_data/data/vocabulary.pkl 3 1024 5 --sample-only --num-names=10 --load-model=model_data/V2/NPLM-w3-emb5-h1024-bs64-lr1e-3-1729.pth --seed=42
```

Output:
```
fannii herdingle colericker fouts-lan
ala raevenammu
widemenwintealerionnife laventonge
saawn
fabbal
wicus
mazouraxid
yphilant grom
fasradami tele pathreigh-hakofrun
blachan
```

In comparision to the n-gram model, this model seems to produce longer names, and most of the names still are difficult to read. However, the names that it generates seem to be more readable than the n-gram model. 

## `train_test_split.py`

Processes the data into training, validation, and testing sets

**Arguments**
- `names_file` :           Path to the CSV file containing NPC names data.
- `save_data_path `:      Path to save the processed data (must be .pkl file).
- `save_vocab_path` :      Path to save the vocabulary (must be .pkl file).
- `window`      :     Context window size (number of previous tokens to consider).
- `--seed SEED `  :        Random seed for reproducibility.
- `--train-size TRAIN_SIZE`: Percentage of samples to use for training (default: 0.8).
- `--val-size VAL_SIZE`:   Percentage of samples to use for validation (default: 0.1).

```bash
python train_test_split.py ../npc_data.csv model_data/data/processed_data_w3.pkl model_data/data/vocabulary.pkl 3 --seed=42
```

## `Embeddings.py`

**Genral Arguments:**
- `vocab_path`: Path to the vocabulary (.pkl file).
- `window`: Context window size (number of previous tokens to consider).
- `emb-size EMB_SIZE`: Size of the character embeddings in the neural network.
- `hidden-size HIDDEN_SIZE`: Size of the hidden layer in the neural network.
- `--data_path DATA_PATH`: Path to the processed datasets (.pkl file).
- `--load_model_path LOAD_MODEL_PATH`: Path to the pre-trained model (with the extension).
- `--save-model-path`: Path to save the trained model (without the extension).
- `--seed SEED`: Random seed for reproducibility.
- `--n_threads N_THREADS`: Number of CPU threads to use.


**Arguments for Sampling:**
- `--sample-only`: Use this flag to generate names without training.
- `--num-names NUM_NAMES`: Number of names to generate (used with --sample-only).
- `--prefix PREFIX`: Optional prefix to start generated names (used with --sample-only).

**Arguments for Training:**
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate for training (default: 0.001).
- `--epochs`: Number of training epochs.
- `--eval-test`: Evaluate the final model on the test set.

**Training Example:**

see `run_model.sh`

```bash
nohup python -u Embeddings.py \
  model_data/data/vocabulary.pkl \
  3 \ # window size
  100 \ # hidden dim
  2 \ # emb size
  --data-path=model_data/data/processed_data_w3.pkl \
  --load-model-path=model_data/model.pth \
  --save-model-path=model_data/model \
  --seed=42 \
  --batch-size=64 \
  --lr=1e-3 \
  --epochs=10
  &
```

(note: -u to avoid output buffering, nohup to run in background since this may take a while)
