import json
import os
import random

import jsonlines
import numpy as np

random.seed(42)


def GetDialog(history, scenario, question):
    dialog = []
    dialog.append(scenario + " " + question)
    for i in history:
        dialog.append(i["follow_up_question"])
        dialog.append(i["follow_up_answer"])
    return dialog


def CreateDataset(dataset, docs={}):
    l = []
    doc_id = len(docs)
    qid = 0

    for example in dataset:
        history = example["history"]
        question = example["question"]
        scenario = example["scenario"]
        answer = example["answer"]
        doc = example["snippet"]
        if (answer == "Irrelevant"):
            answer = "CANNOTANSWER"

        if (doc not in docs):
            docs[doc] = doc_id
            doc_id += 1
        correct_doc_id = docs[doc]

        for i in range(len(history) + 1):
            dialog = GetDialog(history[:i], scenario, question)
            if (i == len(history)):
                response = answer
            else:
                response = history[i]["follow_up_question"]

            d = {
                "qid": qid,
                "doc_id": correct_doc_id,
                "response": response,
                "dialog": dialog
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
    with open("data_sharc/json/sharc_train.json", "r") as f:
        train = json.load(f)

    random.shuffle(train)
    n = int(len(train) * 0.9)
    train, val = train[:n], train[n:]

    with open("data_sharc/json/sharc_dev.json", "r") as f:
        test = json.load(f)

    train, docs = CreateDataset(train)
    val, docs = CreateDataset(val, docs=docs)
    test, docs = CreateDataset(test, docs=docs)

    docs = CreateKnowledge(docs)
    random.shuffle(docs)
    os.makedirs("data_sharc/rag_format")

    with jsonlines.open("data_sharc/rag_format/knowledge.jsonl", "w") as f:
        f.write_all(docs)

    with open("data_sharc/rag_format/train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("data_sharc/rag_format/val.json", "w") as f:
        json.dump(val, f, indent=4)

    with open("data_sharc/rag_format/test.json", "w") as f:
        json.dump(test, f, indent=4)


if (__name__ == "__main__"):
    main()
