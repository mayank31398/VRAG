import random

import jsonlines

random.seed(42)

# ==============================================================================
# create test set
examples = []

with jsonlines.open("data_quac/preprocessed/dev.txt") as f:
    for example in f.iter():
        examples.append(example)

with jsonlines.open("data_quac/preprocessed/test.txt") as f:
    for example in f.iter():
        examples.append(example)

with jsonlines.open("data_quac/augmented/test.txt", "w") as writer:
    writer.write_all(examples)
# ==============================================================================


# ==============================================================================
# create train and dev set
examples = []

with jsonlines.open("data_quac/preprocessed/train.txt") as f:
    for example in f.iter():
        examples.append(example)

print(len(examples))

indices = list(range(len(examples)))
random.shuffle(indices)
n = 10
split_index = len(examples) * n // 100

examples_10 = [examples[i] for i in indices[:split_index]]
examples_90 = [examples[i] for i in indices[split_index:]]

with jsonlines.open("data_quac/augmented/val.txt", "w") as writer:
    writer.write_all(examples_10)

with jsonlines.open("data_quac/augmented/train.txt", "w") as writer:
    writer.write_all(examples_90)
# ==============================================================================
