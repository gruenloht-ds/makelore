Implements a Markov model. The usage is below:

- `names_file`: The provided NPC names data csv file
- `window`: The number of previous characters to consider
- `smoothing_value`: Smoothing value to apply.
- `--num_names NUM_NAMES`: (Optional) Number of names to generate, default = 1
- `--prefix PREFIX`: (Optional) Prefix for generated names to start with, default=""
- `--seed SEED`: (Optional) Seed to generate the same names each time, default=None

Example:

```bash
python MarkovModel.py ../npc_data.csv 3 1 --num_names=10 --seed=42
```
Output
```
gurka
atir ws
daileman
fenrale
maduxtf
bol
kavc
nani
wov
thrag'
```
