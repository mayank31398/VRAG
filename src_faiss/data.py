import json
import logging
import os
import re

logger = logging.getLogger(__name__)

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def remove_articles(_text):
    return RE_ART.sub(' ', _text)


def white_space_fix(_text):
    return ' '.join(_text.split())


def remove_punc(_text):
    return RE_PUNC.sub(' ', _text)  # convert punctuation to spaces


def lower(_text):
    return _text.lower()


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace. """
    return white_space_fix(remove_articles(remove_punc(lower(text))))


def write_preds(eval_dataset, output_file, d, skip_cannot_answer=False):
    l = []
    for example in eval_dataset.prior_dataset.dataset_walker:
        l.append(example)
        if (skip_cannot_answer and example["response"] == "CANNOTANSWER"):
            continue
        l[-1].update(d[example["qid"]])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info("Writing predictions to {}".format(output_file))

    with open(output_file, "w") as jsonfile:
        json.dump(l, jsonfile, indent=4)


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]

    return arrays


def truncate_sequences(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    while words_to_cut > len(sequences[0]):
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]

    sequences[0] = sequences[0][words_to_cut:]
    return sequences
