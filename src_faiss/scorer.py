import argparse
import json
import logging
import re

import jsonlines
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

logger = logging.getLogger(__name__)


class Metrics():
    def __init__(self):
        self._selection_mrrk = []
        self._selection_r1 = []
        self._selection_rk = []
        self._bleu1_from_1_doc = []
        self._bleu2_from_1_doc = []
        self._bleu3_from_1_doc = []
        self._bleu4_from_1_doc = []
        self._ref_cannotanswer_from_1_doc = []
        self._hyp_cannotanswer_from_1_doc = []
        self._bleu1_from_k_docs = []
        self._bleu2_from_k_docs = []
        self._bleu3_from_k_docs = []
        self._bleu4_from_k_docs = []
        self._ref_cannotanswer_from_k_docs = []
        self._hyp_cannotanswer_from_k_docs = []

        self._has_selection_scores = False
        self._has_generation_scores = False

    def _normalize_text(self, text):
        result = text.lower()
        result = RE_PUNC.sub(' ', result)
        result = RE_ART.sub(' ', result)
        result = ' '.join(result.split())

        return result

    def _reciprocal_rank(self, topk_documents_ids, doc_id, k=5):
        mrr = 0
        if (doc_id in topk_documents_ids[:k]):
            r = topk_documents_ids.index(doc_id)
            mrr = 1 / (r + 1)
        return mrr

    def _recall_at_k(self, topk_documents_ids, doc_id, k=5):
        if (doc_id in topk_documents_ids[:k]):
            return 1
        return 0

    def _bleu(self, ref_tokens, hyp_tokens, n=4):
        # if (ref_tokens == ['cannotanswer']):
        #     if (hyp_tokens == ['cannotanswer']):
        #         return 1
        #     else:
        #         return 0
        # else:
        if (hyp_tokens == ['cannotanswer']):
            return 0
        else:
            if (len(hyp_tokens) == 0):
                return 0

            weights = [1 / n] * n
            score = sentence_bleu([ref_tokens], hyp_tokens, weights)

            return score

    def update_selection(self, topk_documents_ids, doc_id):
        self._has_selection_scores = True
        topk = len(topk_documents_ids)

        if (doc_id == None):
            return

        reciprocal_rank = self._reciprocal_rank(
            topk_documents_ids, doc_id, k=topk)
        recall_1 = self._recall_at_k(topk_documents_ids, doc_id, k=1)
        recall_k = self._recall_at_k(topk_documents_ids, doc_id, k=topk)

        self._selection_mrrk.append(reciprocal_rank)
        self._selection_r1.append(recall_1)
        self._selection_rk.append(recall_k)

    def update_generation(self, ref_response, hyp_response, num_docs=1, bleu=-1):
        if (bleu != -1):
            if (num_docs == 1):
                self._bleu1_from_1_doc.append(bleu)
                self._bleu2_from_1_doc.append(bleu)
                self._bleu3_from_1_doc.append(bleu)
                self._bleu4_from_1_doc.append(bleu)
            else:
                self._bleu1_from_k_docs.append(bleu)
                self._bleu2_from_k_docs.append(bleu)
                self._bleu3_from_k_docs.append(bleu)
                self._bleu4_from_k_docs.append(bleu)
            return

        self._has_generation_scores = True
        ref_tokens = self._normalize_text(ref_response).split()
        hyp_tokens = self._normalize_text(hyp_response).split()

        if (num_docs == 1):
            if (ref_tokens == ["cannotanswer"]):
                self._ref_cannotanswer_from_1_doc.append(1)
            else:
                self._ref_cannotanswer_from_1_doc.append(0)

            if (hyp_tokens == ["cannotanswer"]):
                self._hyp_cannotanswer_from_1_doc.append(1)
            else:
                self._hyp_cannotanswer_from_1_doc.append(0)
        else:
            if (ref_tokens == ["cannotanswer"]):
                self._ref_cannotanswer_from_k_docs.append(1)
            else:
                self._ref_cannotanswer_from_k_docs.append(0)

            if (hyp_tokens == ["cannotanswer"]):
                self._hyp_cannotanswer_from_k_docs.append(1)
            else:
                self._hyp_cannotanswer_from_k_docs.append(0)

        if (ref_tokens != ["cannotanswer"]):
            bleu1 = self._bleu(ref_tokens, hyp_tokens, n=1)
            bleu2 = self._bleu(ref_tokens, hyp_tokens, n=2)
            bleu3 = self._bleu(ref_tokens, hyp_tokens, n=3)
            bleu4 = self._bleu(ref_tokens, hyp_tokens, n=4)

            if (num_docs == 1):
                self._bleu1_from_1_doc.append(bleu1)
                self._bleu2_from_1_doc.append(bleu2)
                self._bleu3_from_1_doc.append(bleu3)
                self._bleu4_from_1_doc.append(bleu4)
            else:
                self._bleu1_from_k_docs.append(bleu1)
                self._bleu2_from_k_docs.append(bleu2)
                self._bleu3_from_k_docs.append(bleu3)
                self._bleu4_from_k_docs.append(bleu4)

    def scores(self):
        results = {}
        if (self._has_selection_scores):
            selection_scores = {
                "mrr@k": np.mean(self._selection_mrrk),
                "r@1": np.mean(self._selection_r1),
                "r@k": np.mean(self._selection_rk)
            }
            results.update(selection_scores)
        if (self._has_generation_scores):
            generation_scores = {
                "bleu-1": np.mean(self._bleu1_from_1_doc),
                "bleu-2": np.mean(self._bleu2_from_1_doc),
                "bleu-3": np.mean(self._bleu3_from_1_doc),
                "bleu-4": np.mean(self._bleu4_from_1_doc),
                "accuracy_CANNOTANSWER": accuracy_score(self._ref_cannotanswer_from_1_doc, self._hyp_cannotanswer_from_1_doc),
                "recall_CANNOTANSWER": recall_score(self._ref_cannotanswer_from_1_doc, self._hyp_cannotanswer_from_1_doc),
                "precision_CANNOTANSWER": precision_score(self._ref_cannotanswer_from_1_doc, self._hyp_cannotanswer_from_1_doc),
                "f1_CANNOTANSWER": f1_score(self._ref_cannotanswer_from_1_doc, self._hyp_cannotanswer_from_1_doc)
            }
            results["1_doc"] = generation_scores

            generation_scores = {
                "bleu-1": np.mean(self._bleu1_from_k_docs),
                "bleu-2": np.mean(self._bleu2_from_k_docs),
                "bleu-3": np.mean(self._bleu3_from_k_docs),
                "bleu-4": np.mean(self._bleu4_from_k_docs),
                "accuracy_CANNOTANSWER": accuracy_score(self._ref_cannotanswer_from_k_docs, self._hyp_cannotanswer_from_k_docs),
                "recall_CANNOTANSWER": recall_score(self._ref_cannotanswer_from_k_docs, self._hyp_cannotanswer_from_k_docs),
                "precision_CANNOTANSWER": precision_score(self._ref_cannotanswer_from_k_docs, self._hyp_cannotanswer_from_k_docs),
                "f1_CANNOTANSWER": f1_score(self._ref_cannotanswer_from_k_docs, self._hyp_cannotanswer_from_k_docs)
            }
            results["k_docs"] = generation_scores
        return results


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", type=str)
    parser.add_argument("--knowledge_file", type=str)
    parser.add_argument("--score_file", type=str)
    parser.add_argument("--add_correct_document", action="store_true")
    parser.add_argument("--skip_cannot_answer", action="store_true")
    parser.add_argument("--only_correct_doc", action="store_true")
    parser.add_argument("--penalize", action="store_true")
    args = parser.parse_args()

    predictions = json.load(open(args.output_file, "r"))

    metrics = Metrics()
    for example in tqdm(predictions):
        if (args.skip_cannot_answer and example["response"] == "CANNOTANSWER"):
            continue
        if (args.only_correct_doc and example["doc_id"] != example["topk_documents_ids"][0]):
            continue

        metrics.update_selection(
            example["topk_documents_ids"], example["doc_id"])
        
        if (args.penalize):
            if (example["doc_id"] == example["topk_documents_ids"][0]):
                metrics.update_generation(
                    example["response"], example["generated_response_from_1_doc"][0])
            else:
                metrics.update_generation(None, None, bleu=0)
            
            if (example["doc_id"] in example["topk_documents_ids"]):
                metrics.update_generation(
                    example["response"], example["generated_response_from_k_docs"], num_docs="k")
            else:
                metrics.update_generation(None, None, num_docs="k", bleu=0)
        else:
            metrics.update_generation(
                example["response"], example["generated_response_from_1_doc"][0])
            metrics.update_generation(
                example["response"], example["generated_response_from_k_docs"], num_docs="k")

    results = metrics.scores()

    logger.info("***** Results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    if (args.add_correct_document):
        knowledge = {}
        with jsonlines.open(args.knowledge_file, "r") as f:
            for i in f.iter():
                knowledge[i["id"]] = i
                del knowledge[i["id"]]["id"]

        for i in range(len(predictions)):
            if (args.skip_cannot_answer and predictions[i]["response"] == "CANNOTANSWER"):
                continue

            if (predictions[i]["doc_id"] != None):
                predictions[i]["correct_document"] = knowledge[predictions[i]["doc_id"]]
            else:
                predictions[i]["correct_document"] = None

            predictions[i]["topk_documents"] = []
            for j in range(len(predictions[i]["topk_documents_ids"])):
                predictions[i]["topk_documents"].append(
                    knowledge[predictions[i]["topk_documents_ids"][j]])

        json.dump(predictions, open(args.output_file, "w"), indent=4)

    json.dump(results, open(args.score_file, "w"), indent=4)


if (__name__ == "__main__"):
    main()
