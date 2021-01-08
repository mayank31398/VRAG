import json
import os
import random

import jsonlines
import numpy as np

random.seed(42)


def CreateDataset(dataset, docs={}, qid=0):
    l = []
    qid = 0

    for key in dataset["query"]:
        evidences = dataset["passages"][key]
        question = dataset["query"][key]
        answer = dataset["answers"][key][0]
        correct_doc_id = None

        if (answer.lower() == "no answer present."):
            answer = "CANNOTANSWER"

        e = []
        for doc_id, evidence in enumerate(evidences):
            t = {
                "title": "",
                "text": evidence["passage_text"]
            }
            e.append(t)

            if (evidence["is_selected"] == 1):
                correct_doc_id = doc_id

        d = {
            "qid": qid,
            "doc_id": correct_doc_id,
            "query": question,
            "response": answer,
            "docs": e
        }
        l.append(d)
        qid += 1

    return l


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
    with open("data_marco/augmented/train.json", "r") as f:
        train = json.load(f)
    train = CreateDataset(train)

    with open("data_marco/augmented/val.json", "r") as f:
        val = json.load(f)
    val = CreateDataset(val)

    with open("data_marco/augmented/test.json", "r") as f:
        test = json.load(f)
    test = CreateDataset(test)

    os.makedirs("data_marco/close_format")

    with open("data_marco/close_format/train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("data_marco/close_format/val.json", "w") as f:
        json.dump(val, f, indent=4)

    with open("data_marco/close_format/test.json", "w") as f:
        json.dump(test, f, indent=4)


if (__name__ == "__main__"):
    main()
