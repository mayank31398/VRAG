import argparse
import json
import logging
import re
import sys

import jsonlines
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

logger = logging.getLogger(__name__)


class Metrics():
    def __init__(self, topk=None):
        self._selection_mrrk = []
        self._selection_r1 = []
        self._selection_rk = []
        self._bleu1 = []
        self._bleu2 = []
        self._bleu3 = []
        self._bleu4 = []
        self._ref_cannotanswer = []
        self._hyp_cannotanswer = []
        self.topk = topk

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
        if (ref_tokens == ['cannotanswer']):
            if (hyp_tokens == ['cannotanswer']):
                return 1
            else:
                return 0
        else:
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
        if (self.topk == None):
            self.topk = len(topk_documents_ids)

        if (doc_id == None):
            return

        reciprocal_rank = self._reciprocal_rank(
            topk_documents_ids, doc_id, k=self.topk)
        recall_1 = self._recall_at_k(topk_documents_ids, doc_id, k=1)
        recall_k = self._recall_at_k(topk_documents_ids, doc_id, k=self.topk)

        self._selection_mrrk.append(reciprocal_rank)
        self._selection_r1.append(recall_1)
        self._selection_rk.append(recall_k)

    def update_generation(self, ref_response, hyp_response):
        self._has_generation_scores = True
        ref_tokens = self._normalize_text(ref_response).split()
        hyp_tokens = self._normalize_text(hyp_response).split()

        if (ref_tokens == ["cannotanswer"]):
            self._ref_cannotanswer.append(1)
        else:
            self._ref_cannotanswer.append(0)

        if (hyp_tokens == ["cannotanswer"]):
            self._hyp_cannotanswer.append(1)
        else:
            self._hyp_cannotanswer.append(0)

        bleu1 = self._bleu(ref_tokens, hyp_tokens, n=1)
        bleu2 = self._bleu(ref_tokens, hyp_tokens, n=2)
        bleu3 = self._bleu(ref_tokens, hyp_tokens, n=3)
        bleu4 = self._bleu(ref_tokens, hyp_tokens, n=4)

        self._bleu1.append(bleu1)
        self._bleu2.append(bleu2)
        self._bleu3.append(bleu3)
        self._bleu4.append(bleu4)

    def scores(self):
        results = {}
        if (self._has_selection_scores):
            selection_scores = {
                "mrr@" + str(self.topk): np.mean(self._selection_mrrk),
                "r@1": np.mean(self._selection_r1),
                "r@" + str(self.topk): np.mean(self._selection_rk)
            }
            results.update(selection_scores)
        if (self._has_generation_scores):
            generation_scores = {
                "bleu-1": np.mean(self._bleu1),
                "bleu-2": np.mean(self._bleu2),
                "bleu-3": np.mean(self._bleu3),
                "bleu-4": np.mean(self._bleu4),
                "accuracy_CANNOTANSWER": accuracy_score(self._ref_cannotanswer, self._hyp_cannotanswer),
                "recall_CANNOTANSWER": recall_score(self._ref_cannotanswer, self._hyp_cannotanswer),
                "precision_CANNOTANSWER": precision_score(self._ref_cannotanswer, self._hyp_cannotanswer),
                "f1_CANNOTANSWER": f1_score(self._ref_cannotanswer, self._hyp_cannotanswer)
            }
            results.update(generation_scores)
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
    args = parser.parse_args()

    predictions = json.load(open(args.output_file, "r"))

    metrics = Metrics()
    for example in predictions:
        metrics.update_selection(
            example["topk_documents_ids"], example["doc_id"])
        metrics.update_generation(
            example["response"], example["generated_response_from_1_doc"])
    results = metrics.scores()

    logger.info("***** Results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    knowledge = {}
    with jsonlines.open(args.knowledge_file, "r") as f:
        for i in f.iter():
            knowledge[i["id"]] = i
            del knowledge[i["id"]]["id"]

    for i in range(len(predictions)):
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
