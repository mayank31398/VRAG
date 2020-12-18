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

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

logger = logging.getLogger(__name__)


class SelectionMetrics:
    def __init__(self, topk=None):
        self.reset()
        self.topk = topk

    def reset(self):
        self._count = 0
        self._selection_mrrk = 0
        self._selection_r1 = 0
        self._selection_rk = 0

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

    def update(self, topk_documents_ids, doc_id):
        if (self.topk == None):
            self.topk = len(topk_documents_ids)

        if (doc_id == None):
            return None, None, None

        reciprocal_rank = self._reciprocal_rank(
            topk_documents_ids, doc_id, k=self.topk)
        recall_1 = self._recall_at_k(topk_documents_ids, doc_id, k=1)
        recall_k = self._recall_at_k(topk_documents_ids, doc_id, k=self.topk)

        self._selection_mrrk += reciprocal_rank
        self._selection_r1 += recall_1
        self._selection_rk += recall_k
        self._count += 1

        return reciprocal_rank, recall_1, recall_k

    def _compute(self, score_sum):
        return score_sum / self._count

    def scores(self):
        selection_mrrk_f = self._compute(self._selection_mrrk)
        selection_r1_f = self._compute(self._selection_r1)
        selection_rk_f = self._compute(self._selection_rk)

        scores = {
            "mrr@" + str(self.topk): selection_mrrk_f,
            "r@1": selection_r1_f,
            "r@" + str(self.topk): selection_rk_f,
        }
        return scores


class GenerationMetrics:
    def __init__(self, topk=None):
        self.reset()

    def reset(self):
        self._count = 0
        self._bleu1 = 0
        self._bleu2 = 0
        self._bleu3 = 0
        self._bleu4 = 0

    def _normalize_text(self, text):
        result = text.lower()
        result = RE_PUNC.sub(' ', result)
        result = RE_ART.sub(' ', result)
        result = ' '.join(result.split())

        return result

    def _bleu(self, ref_response, hyp_response, n=4):
        ref_tokens = self._normalize_text(ref_response).split()
        hyp_tokens = self._normalize_text(hyp_response).split()

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

    def update(self, ref_response, hyp_response):
        bleu1 = self._bleu(ref_response, hyp_response, n=1)
        bleu2 = self._bleu(ref_response, hyp_response, n=2)
        bleu3 = self._bleu(ref_response, hyp_response, n=3)
        bleu4 = self._bleu(ref_response, hyp_response, n=4)

        self._bleu1 += bleu1
        self._bleu2 += bleu2
        self._bleu3 += bleu3
        self._bleu4 += bleu4
        self._count += 1

        return bleu1, bleu2, bleu3, bleu4

    def _compute(self, score_sum):
        return score_sum / self._count

    def scores(self):
        bleu1 = self._compute(self._bleu1)
        bleu2 = self._compute(self._bleu2)
        bleu3 = self._compute(self._bleu3)
        bleu4 = self._compute(self._bleu4)

        scores = {
            "bleu-1": bleu1,
            "bleu-2": bleu2,
            "bleu-3": bleu3,
            "bleu-4": bleu4
        }
        return scores


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

    selection_metrics = SelectionMetrics()
    for example in predictions:
        selection_metrics.update(
            example["topk_documents_ids"], example["doc_id"])
    results_selection = selection_metrics.scores()

    logger.info("***** Selection results *****")
    for key in sorted(results_selection.keys()):
        logger.info("  %s = %s", key, str(results_selection[key]))

    generation_metrics = GenerationMetrics()
    for example in predictions:
        generation_metrics.update(
            example["response"], example["generated_response_from_1_doc"])
    results_generation = generation_metrics.scores()

    logger.info("***** Generation results *****")
    for key in sorted(results_generation.keys()):
        logger.info("  %s = %s", key, str(results_generation[key]))

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

    scores = {
        "selection": results_selection,
        "generation": results_generation
    }
    json.dump(scores, open(args.score_file, "w"), indent=4)


if (__name__ == "__main__"):
    main()
