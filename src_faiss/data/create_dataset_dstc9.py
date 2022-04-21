import json
import os
import random

import jsonlines

random.seed(42)


def GetKey(domain, entity_id, doc_id):
    return domain + "__" + str(entity_id) + "__" + str(doc_id)


def CreateDataset(logs, labels, key_map):
    qid = 0
    l = []
    for i in range(len(labels)):
        if (not labels[i]["target"]):
            continue

        dialog = []
        for j in logs[i]:
            dialog.append(j["text"])

        domain = labels[i]["knowledge"][0]["domain"]
        entity_id = labels[i]["knowledge"][0]["entity_id"]
        doc_id = labels[i]["knowledge"][0]["doc_id"]
        key = GetKey(domain, entity_id, doc_id)
        correct_doc_id = key_map[key]

        d = {
            "qid": qid,
            "doc_id": correct_doc_id,
            "response": labels[i]["response"],
            "dialog": dialog
        }
        l.append(d)
        qid += 1

    return l


def CreateKnowledge(docs, key_map):
    docs_ = []
    for i in key_map:
        d = {
            "id": key_map[i],
            "title": "",
            "text": docs[key_map[i]]
        }
        docs_.append(d)
    return docs_


def main():
    docs = {}
    key_map = {}
    i = 0
    with open("data_dstc/data_eval/knowledge.json", "r") as f:
        knowledge = json.load(f)
        for domain in knowledge:
            for entity_id in knowledge[domain]:
                entity_name = knowledge[domain][entity_id]["name"]
                if (entity_name == None):
                    entity_name = domain
                for doc_id in knowledge[domain][entity_id]["docs"]:
                    q = knowledge[domain][entity_id]["docs"][doc_id]["title"]
                    a = knowledge[domain][entity_id]["docs"][doc_id]["body"]

                    key_map[GetKey(domain, entity_id, doc_id)] = i
                    docs[i] = entity_name + " : " + q + " : " + a
                    i += 1

    with open("data_dstc/data/train/logs.json", "r") as f:
        train_logs = json.load(f)
    with open("data_dstc/data/train/labels.json", "r") as f:
        train_labels = json.load(f)
    train = CreateDataset(train_logs, train_labels, key_map)

    with open("data_dstc/data/val/logs.json", "r") as f:
        val_logs = json.load(f)
    with open("data_dstc/data/val/labels.json", "r") as f:
        val_labels = json.load(f)
    val = CreateDataset(val_logs, val_labels, key_map)

    with open("data_dstc/data_eval/test/logs.json", "r") as f:
        test_logs = json.load(f)
    with open("data_dstc/data_eval/test/labels.json", "r") as f:
        test_labels = json.load(f)
    test = CreateDataset(test_logs, test_labels, key_map)

    docs = CreateKnowledge(docs, key_map)
    random.shuffle(docs)

    os.makedirs("data_dstc/rag_format")

    with jsonlines.open("data_dstc/rag_format/knowledge.jsonl", "w") as f:
        f.write_all(docs)

    with open("data_dstc/rag_format/train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("data_dstc/rag_format/val.json", "w") as f:
        json.dump(val, f, indent=4)

    with open("data_dstc/rag_format/test.json", "w") as f:
        json.dump(test, f, indent=4)


if (__name__ == "__main__"):
    main()
