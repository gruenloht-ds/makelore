Splits data into training, validation, and testing sets for comparison across models.

To run:

```bash

python make_split.py \
--data-path=~/npc_data.csv \
--save-path=~/data/ \
--train-size=0.8 \
--test-size=0.1 \
--seed=42

```
