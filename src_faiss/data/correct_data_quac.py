import jsonlines
import json

x = json.load(open("tmp/train.json", "r"))["data"]
x += json.load(open("tmp/val.json", "r"))["data"]

d = {}
for i in x:
    for k in i["paragraphs"]:
        for j in k["qas"]:
            d[j["id"]] = k["context"].split("CANNOTANSWER")[0]

l = []
with jsonlines.open("data_quac/preprocessed/dev.txt", "r") as f:
    for i in f.iter():
        qid = i["qid"]

        if (sum(i["retrieval_labels"]) == 0):
            i["evidences"].append(d[qid])
            i["retrieval_labels"].append(1)

        l.append(i)

for i in l:
    if (sum(i["retrieval_labels"]) == 0):
        print("NO")

with jsonlines.open("tmp/dev1.txt", "w") as writer:
    writer.write_all(l)