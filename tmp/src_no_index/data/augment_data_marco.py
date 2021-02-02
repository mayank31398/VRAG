import copy
import json
import os
import random

random.seed(42)

# ==============================================================================
# create test set
with open("data_marco/dev_v2.1.json") as f:
    examples = json.load(f)

print(len(examples["query"]))
os.makedirs("data_marco/augmented")

with open("data_marco/augmented/test.json", "w") as f:
    json.dump(examples, f, indent=4)
# ==============================================================================


# ==============================================================================
# create train and dev set
with open("data_marco/train_v2.1.json") as f:
    examples = json.load(f)

print(len(examples["query"]))

keys = list(examples["query"].keys())
random.shuffle(keys)

n = 10
split_index = len(keys) * n // 100

examples1 = copy.deepcopy(examples)

keys_10 = keys[:split_index]
keys_90 = keys[split_index:]

for i in keys_10:
    for k in examples:
        del examples[k][i]

for i in keys_90:
    for k in examples1:
        del examples1[k][i]

with open("data_marco/augmented/val.json", "w") as f:
    json.dump(examples1, f, indent=4)

with open("data_marco/augmented/train.json", "w") as f:
    json.dump(examples, f, indent=4)
# ==============================================================================
