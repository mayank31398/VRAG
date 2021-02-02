import copy
import json
import os
import random

import jsonlines
import numpy as np
from tqdm import tqdm

random.seed(42)


def CreateDataset(dataset, docs={}, qid=0, write_candidate_responses=False):
    l = []
    doc_id = len(docs)
    qid = 0

    for e in tqdm(dataset):
        i = e["dialog"]
        dialog = []
        j = 1
        if ("Wizard" not in i[0]["speaker"]):
            dialog.append(i[j - 1]["text"])
            j += 1

        while (j < len(i)):
            dialog.append(i[j - 1]["text"])
            response = i[j]["text"]
            if (write_candidate_responses):
                candidate_responses = i[j - 1]["candidate_responses"]
            if (len(list(i[j - 1]["checked_sentence"].values())) == 0):
                dialog.append(response)
                j += 2
                continue

            correct_doc = list(i[j - 1]["checked_sentence"].values())[0]

            for k in i:
                k = k["retrieved_passages"]
                for k_ in k:
                    k_ = list(k_.values())[0]
                    for doc_ in k_:
                        if (doc_ not in docs):
                            docs[doc_] = doc_id
                            doc_id += 1

            if (correct_doc not in docs):
                docs[correct_doc] = doc_id
                doc_id += 1

            correct_doc_id = docs[correct_doc]

            d = {
                "qid": qid,
                "doc_id": correct_doc_id,
                "response": response,
                "dialog": copy.deepcopy(dialog)
            }
            if (write_candidate_responses):
                d["candidate_responses"] = i[j - 1]["candidate_responses"]
            l.append(d)
            qid += 1
            j += 2

            dialog.append(response)

    return l, docs


def CreateKnowledge(docs):
    l = []
    for evidence in docs:
        doc_id = docs[evidence]
        d = {
            "id": doc_id,
            "title": "",
            "text": evidence
        }
        l.append(d)

    return l


def main():
    with open("data_wow/download/train.json", "r") as f:
        train = json.load(f)
    train, docs = CreateDataset(train)

    with open("data_wow/download/valid_random_split.json", "r") as f:
        val = json.load(f)
    val, docs = CreateDataset(val, docs=docs, write_candidate_responses=True)

    with open("data_wow/download/test_random_split.json", "r") as f:
        test = json.load(f)
    test, docs = CreateDataset(test, docs=docs, write_candidate_responses=True)

    docs = CreateKnowledge(docs)
    random.shuffle(docs)

    os.makedirs("data_wow/rag_format")

    with jsonlines.open("data_wow/rag_format/knowledge.jsonl", "w") as f:
        f.write_all(docs)

    with open("data_wow/rag_format/train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("data_wow/rag_format/val.json", "w") as f:
        json.dump(val, f, indent=4)

    with open("data_wow/rag_format/test.json", "w") as f:
        json.dump(test, f, indent=4)


if (__name__ == "__main__"):
    main()
