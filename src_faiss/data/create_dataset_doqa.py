import copy
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

    for i in dataset:
        i = i["paragraphs"][0]
        correct_doc = i["context"].split("CANNOTANSWER")[0].strip()

        if (correct_doc not in docs):
            docs[correct_doc] = doc_id
            doc_id += 1

        correct_doc_id = docs[correct_doc]
        dialog = []

        j = 0
        while (j < len(i["qas"])):
            dialog.append(i["qas"][j]["question"])
            answer = i["qas"][j]["answers"][0]["text"]

            d = {
                "qid": qid,
                "doc_id": correct_doc_id,
                "response": answer,
                "dialog": copy.deepcopy(dialog)
            }
            l.append(d)
            qid += 1
            j += 1

            dialog.append(answer)

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
    with open("data_doqa/doqa-v2.1/doqa_dataset/doqa-cooking-train-v2.1.json", "r") as f:
        train = json.load(f)["data"]
    train, docs = CreateDataset(train)

    with open("data_doqa/doqa-v2.1/doqa_dataset/doqa-cooking-dev-v2.1.json", "r") as f:
        val = json.load(f)["data"]
    val, docs = CreateDataset(val, docs=docs)

    with open("data_doqa/doqa-v2.1/doqa_dataset/doqa-cooking-test-v2.1.json", "r") as f:
        test = json.load(f)["data"]
    test, docs = CreateDataset(test, docs=docs)

    docs = CreateKnowledge(docs)
    random.shuffle(docs)

    os.makedirs("data_doqa/rag_format")

    with jsonlines.open("data_doqa/rag_format/knowledge.jsonl", "w") as f:
        f.write_all(docs)

    with open("data_doqa/rag_format/train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("data_doqa/rag_format/val.json", "w") as f:
        json.dump(val, f, indent=4)

    with open("data_doqa/rag_format/test.json", "w") as f:
        json.dump(test, f, indent=4)


if (__name__ == "__main__"):
    main()
