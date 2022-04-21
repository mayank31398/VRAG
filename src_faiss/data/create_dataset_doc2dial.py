import copy
import json
import os
import random

import jsonlines

random.seed(42)


def GetDialog(example):
    l = []
    dialog = []
    role = "user"
    i = 0
    while (i < len(example["turns"]) - 1):
        if (example["turns"][i]["role"] == role):
            dialog.append(example["turns"][i]["utterance"])
        else:
            dialog[-1] = dialog[-1] + " " + example["turns"][i]["utterance"]
        i += 1

        if (example["turns"][i]["role"] != role):
            response = example["turns"][i]["utterance"]
            l.append((copy.deepcopy(dialog), response))

        if (role == "user"):
            role = "agent"
        else:
            role = "user"
    return l


def CreateDataset(dataset, docs={}):
    l = []
    doc_id = len(docs)
    qid = 0

    for example in dataset:
        doc_hash = example["doc_id"]
        if (doc_hash not in docs):
            docs[doc_hash] = doc_id
            doc_id += 1
        correct_doc_id = docs[doc_hash]

        q = GetDialog(example)
        for dialog, response in q:
            d = {
                "qid": qid,
                "doc_id": correct_doc_id,
                "response": response,
                "dialog": dialog
            }
            l.append(d)
            qid += 1

    return l, docs


def CreateKnowledge(docs, knowledge):
    q = {}
    for domain in knowledge["doc_data"]:
        for doc_hash in knowledge["doc_data"][domain]:
            q[doc_hash] = knowledge["doc_data"][domain][doc_hash]["doc_text"]

    l = []
    for doc_hash in docs:
        doc_id = docs[doc_hash]
        d = {
            "id": doc_id,
            "title": "",
            "text": q[doc_hash]
        }
        l.append(d)

    return l

def Unravel(dataset):
    l = []
    for domain in dataset["dial_data"]:
        for doc_hash in dataset["dial_data"][domain]:
            for i in dataset["dial_data"][domain][doc_hash]:
                l.append(i)
    return l

def main():
    with open("data_doc2dial/v1.0.1/doc2dial_dial_train.json", "r") as f:
        train = json.load(f)
    train = Unravel(train)

    with open("data_doc2dial/v1.0.1/doc2dial_dial_validation.json", "r") as f:
        test = json.load(f)
    test = Unravel(test)

    random.shuffle(train)
    n = int(len(train) * 0.9)
    train, val = train[:n], train[n:]

    train, docs = CreateDataset(train)
    val, docs = CreateDataset(val, docs=docs)
    test, docs = CreateDataset(test, docs=docs)

    with open("data_doc2dial/v1.0.1/doc2dial_doc.json", "r") as f:
        knowledge = json.load(f)

    docs = CreateKnowledge(docs, knowledge)
    random.shuffle(docs)

    os.makedirs("data_doc2dial/rag_format")

    with jsonlines.open("data_doc2dial/rag_format/knowledge.jsonl", "w") as f:
        f.write_all(docs)

    with open("data_doc2dial/rag_format/train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("data_doc2dial/rag_format/val.json", "w") as f:
        json.dump(val, f, indent=4)

    with open("data_doc2dial/rag_format/test.json", "w") as f:
        json.dump(test, f, indent=4)


if (__name__ == "__main__"):
    main()
