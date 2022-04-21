import json
import os
import random

import jsonlines
import numpy as np

random.seed(42)


def GetDialog(history, question):
    l = []
    for i in history:
        l.append(i["question"])
        l.append(i["answer"]["text"])
    l.append(question)
    return l


def CreateDataset(dataset, docs={}, qid=0):
    l = []
    doc_id = len(docs)

    for example in dataset:
        evidences = example["evidences"]
        history = example["history"]
        question = example["question"]
        rewrite = example["rewrite"]
        answer = example["answer"]["text"]
        retrieval_labels = example["retrieval_labels"]

        for evidence in evidences:
            if (evidence not in docs):
                docs[evidence] = doc_id
                doc_id += 1

        if (sum(retrieval_labels) == 0):
            correct_doc_id = None
        else:
            label_idx = np.argmax(retrieval_labels)
            correct_evidence = evidences[label_idx]
            correct_doc_id = docs[correct_evidence]

        d = {
            "qid": qid,
            "doc_id": correct_doc_id,
            "query": rewrite,
            "response": answer,
            "dialog": GetDialog(history, question)
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
    train = []
    with jsonlines.open("data_quac/preprocessed/train.txt", "r") as f:
        for example in f.iter():
            train.append(example)
    train, docs = CreateDataset(train)

    val = []
    with jsonlines.open("data_quac/preprocessed/dev.txt", "r") as f:
        for example in f.iter():
            val.append(example)
    val, docs = CreateDataset(val, docs=docs)

    test = []
    with jsonlines.open("data_quac/preprocessed/test.txt", "r") as f:
        for example in f.iter():
            test.append(example)
    test, docs = CreateDataset(test, docs=docs)

    docs = CreateKnowledge(docs)
    random.shuffle(docs)
    os.makedirs("data_quac/rag_format")

    with jsonlines.open("data_quac/rag_format/knowledge.jsonl", "w") as f:
        f.write_all(docs)

    with open("data_quac/rag_format/train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("data_quac/rag_format/val.json", "w") as f:
        json.dump(val, f, indent=4)

    with open("data_quac/rag_format/test.json", "w") as f:
        json.dump(test, f, indent=4)


if (__name__ == "__main__"):
    main()
