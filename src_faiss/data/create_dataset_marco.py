import json
import os
import random

import jsonlines
import numpy as np

random.seed(42)


def CreateDataset(dataset, docs={}):
    l = []
    doc_id = len(docs)
    qid = 0

    for key in dataset["query"]:
        evidences = dataset["passages"][key]
        question = dataset["query"][key]
        answer = dataset["answers"][key][0]
        correct_doc_id = None

        if (answer.lower() == "no answer present."):
            answer = "CANNOTANSWER"

        for evidence in evidences:
            if (evidence["passage_text"] not in docs):
                docs[evidence["passage_text"]] = doc_id
                doc_id += 1

                if (evidence["is_selected"] == 1):
                    correct_doc_id = doc_id

        d = {
            "qid": qid,
            "doc_id": correct_doc_id,
            "query": question,
            "response": answer
        }
        l.append(d)
        qid += 1

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
    with open("data_marco/augmented/train.json", "r") as f:
        train = json.load(f)
    train, docs = CreateDataset(train)

    with open("data_marco/augmented/val.json", "r") as f:
        val = json.load(f)
    val, docs = CreateDataset(val, docs=docs)

    with open("data_marco/augmented/test.json", "r") as f:
        test = json.load(f)
    test, docs = CreateDataset(test, docs=docs)

    docs = CreateKnowledge(docs)
    random.shuffle(docs)

    os.makedirs("data_marco/rag_format")

    with jsonlines.open("data_marco/rag_format/knowledge.jsonl", "w") as f:
        f.write_all(docs)

    with open("data_marco/rag_format/train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("data_marco/rag_format/val.json", "w") as f:
        json.dump(val, f, indent=4)

    with open("data_marco/rag_format/test.json", "w") as f:
        json.dump(test, f, indent=4)


if (__name__ == "__main__"):
    main()
