import argparse
import json
import re
import sys
import logging
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    predictions = json.load(open(args.output_file, "r"))

    selection_metrics = SelectionMetrics()
    for example in predictions:
        selection_metrics.update(example["topk_documents_ids"], example["doc_id"])
    results = selection_metrics.scores()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))


if (__name__ == "__main__"):
    main()
