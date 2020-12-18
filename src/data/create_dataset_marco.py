import json
import random

import jsonlines
import numpy as np

random.seed(42)


def CreateDataset(dataset, docs={}, qid=0):
    l = []
    doc_id = len(docs)
    qid = 0

    for key in dataset["query"]:
        evidences = dataset["passages"][key]
        question = dataset["query"][key]

        if ("answers" in dataset):
            answer = dataset["answers"][key][0]
            if (answer.lower() == "no answer present."):
                answer = "CANNOTANSWER"
        else:
            answer = None

        for evidence in evidences:
            if (evidence["passage_text"] not in docs):
                docs[evidence["passage_text"]] = doc_id
                doc_id += 1

        d = {
            "qid": qid,
            "doc_id": None,
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
    with open("data_marco/train.json", "r") as f:
        train = json.load(f)
    train, docs = CreateDataset(train)

    with open("data_marco/val.json", "r") as f:
        val = json.load(f)
    val, docs = CreateDataset(val, docs=docs)

    with open("data_marco/test.json", "r") as f:
        test = json.load(f)
    test, docs = CreateDataset(test, docs=docs)

    docs = CreateKnowledge(docs)
    random.shuffle(docs)

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
