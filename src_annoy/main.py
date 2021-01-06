import argparse
import copy
import json
import logging
import os
import random
from argparse import Namespace

import annoy
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (AdamW, DPRContextEncoder, DPRContextEncoderTokenizer,
                          get_linear_schedule_with_warmup)

from .data import write_preds
from .dataset import (DecoderDataset, PosteriorDataset, PriorDataset,
                      UnsupervisedDataset)
from .models import DecoderModel, PosteriorModel, PriorModel, UnsupervisedModel
from .scorer import Metrics

logger = logging.getLogger(__name__)


def embed(documents, ctx_encoder, ctx_tokenizer):
    input_ids = ctx_tokenizer(documents["title"], documents["text"],
                              truncation=True, padding="longest", return_tensors="pt")["input_ids"]
    embeddings = ctx_encoder(input_ids.cuda(), return_dict=True).pooler_output
    return embeddings.detach().cpu().numpy()


class KnowledgeWalker:
    def __init__(self, args):
        self.index = annoy.AnnoyIndex(
            args.index_args["dimensions"], metric="euclidean")

        self.dataset = []
        with jsonlines.open(args.knowledge_file, "r") as f:
            for i in f.iter():
                self.dataset.append(i)

        if (args.build_index):
            ctx_encoder = DPRContextEncoder.from_pretrained(
                args.document_encoder_model_name).cuda()
            ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                args.document_encoder_model_name)

            self.vectors = []
            for i in tqdm(self.dataset):
                v = embed(i, ctx_encoder, ctx_tokenizer)
                self.vectors.append(v)
            self.vectors = np.concatenate(self.vectors)

            for i, vec in enumerate(self.vectors):
                self.index.add_item(i, vec.tolist())
            self.index.build(args.index_args["num_trees"])

            os.makedirs(args.index_path, exist_ok=True)
            np.save(os.path.join(args.index_path, "vectors.npy"), self.vectors)
            self.index.save(os.path.join(args.index_path, "index.annoy"))

            exit()
        else:
            self.vectors = np.load(os.path.join(
                args.index_path, "vectors.npy"))
            self.index.load(os.path.join(args.index_path, "index.annoy"))

    def retrieve(self, batch, k=5):
        indices = []
        for vec in batch:
            indices.append(self.get_indices(vec, k))
        return indices

    def get_indices(self, vector, k):
        return self.index.get_nns_by_vector(vector.tolist(), k)

    def get_vectors_by_indices(self, indices):
        return [self.vectors[i] for i in indices]

    def get_data_by_indices(self, indices):
        return [self.dataset[i] for i in indices]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def Train(args, train_dataset, eval_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(
        train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    model.zero_grad()
    global_step = 0
    best_acc = 0
    num_times_best_acc = 0
    best_found = False

    results = Evaluate(args, eval_dataset, model)

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        local_steps = 0
        tr_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        skip_counter = 0

        for step, batch in enumerate(epoch_iterator):
            global_step += 1

            # with torch.autograd.detect_anomaly():
            loss = model(batch)
            if (torch.isnan(loss).sum() >= 1):
                skip_counter += 1
                print("skipped =", skip_counter)
                continue

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            if ((step + 1) % args.gradient_accumulation_steps == 0):
                torch.nn.utils.clip_grad_norm_(
                    model.GetParameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss / (local_steps + 1))

        results = Evaluate(args, eval_dataset, model)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        if (results["r@1"] > best_acc):
            num_times_best_acc = 0
            model.save_model(args, "best")
            best_acc = results["r@1"]
            best_found = True
        else:
            num_times_best_acc += 1
            if (num_times_best_acc == args.stopping_criteria):
                break

        model.save_model(args, "checkpoint-" + str(global_step))

    model.save_model(args, "")
    if (not best_found):
        model.save_model(args, "best")


def Evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,
        collate_fn=eval_dataset.collate_fn
    )

    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    metrics = Metrics()

    d = {}
    with torch.no_grad():
        model.eval()

        for batch in epoch_iterator:
            prior_input_ids, _, _, decoder_input_ids, _, doc_ids, q_ids = batch

            prior_indices, prior_logits, _, _ = model.prior_model(
                prior_input_ids.cuda(), args.topk)

            prior_indices = prior_indices[0]

            if (args.n_gpus > 1):
                prior_data = model.prior_model.module.indexed_passages.get_data_by_indices(
                    prior_indices)
            else:
                prior_data = model.prior_model.indexed_passages.get_data_by_indices(
                    prior_indices)

            prior_logits = prior_logits[0]
            prior_dist = F.softmax(
                prior_logits, dim=-1).detach().cpu().numpy().tolist()

            decoder_input_ids = decoder_input_ids[0]

            prior_topk_documents_ids = [i["id"] for i in prior_data]
            prior_topk_documents_text = [i["text"] for i in prior_data]
            doc_ids = doc_ids[0]
            q_ids = q_ids[0]

            metrics.update_selection(prior_topk_documents_ids, doc_ids)

            if (args.eval_only):
                if (args.n_gpus > 1):
                    output_text_from_1_doc = model.decoder_model.module.generate_from_1_doc(
                        args, decoder_input_ids, prior_topk_documents_text[0])
                    output_text_from_k_docs = model.decoder_model.module.generate_from_k_docs(
                        args, decoder_input_ids, prior_topk_documents_text, prior_dist)
                else:
                    output_text_from_1_doc = model.decoder_model.generate_from_1_doc(
                        args, decoder_input_ids, prior_topk_documents_text[0])
                    output_text_from_k_docs = model.decoder_model.generate_from_k_docs(
                        args, decoder_input_ids, prior_topk_documents_text, prior_dist)

                d[q_ids] = {
                    "prior_dist": prior_dist,
                    "topk_documents_ids": prior_topk_documents_ids,
                    "generated_response_from_1_doc": output_text_from_1_doc,
                    "generated_response_from_k_docs": output_text_from_k_docs
                }

    if (args.eval_only):
        write_preds(eval_dataset, args.output_file, d)

    results = metrics.scores()
    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str,
                        help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Saved checkpoint directory")
    parser.add_argument("--dataroot", type=str, help="Path to dataset.")
    parser.add_argument("--knowledge_file", type=str,
                        help="Path to knowledge file.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead.")
    parser.add_argument("--output_file", type=str, default="",
                        help="Predictions will be written to this file.")
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument("--model_path", type=str,
                        help="Name of the experiment, checkpoints will be stored here")
    parser.add_argument(
        "--build_index", action="store_true", help="Build index")
    parser.add_argument("--index_path", type=str, help="Path of the index")
    parser.add_argument("--n_gpus", type=int, default=1, help="Num GPUS")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # load args from params file and update the args Namespace
    if (args.eval_only and args.params_file == None):
        args.params_file = os.path.join(
            args.model_path, "prior", args.checkpoint, "params.json")
    logger.info("using params from " + args.params_file)

    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        args.update(params)
        args = Namespace(**args)

        for key in vars(args):
            logger.info(str(key) + " = " + str(vars(args)[key]))

    args.params = params  # used for saving checkpoints

    # Set seed
    set_seed(args)

    indexed_passages = KnowledgeWalker(args)

    args.batch_size = args.batch_size * args.n_gpus
    unsupervised_model = UnsupervisedModel(args, indexed_passages).cuda()

    if (args.n_gpus > 1):
        tokenizers = {
            "prior_tokenizer": unsupervised_model.prior_model.module.tokenizer,
            "posterior_tokenizer": unsupervised_model.posterior_model.module.tokenizer,
            "decoder_tokenizer": unsupervised_model.decoder_model.module.tokenizer
        }
    else:
        tokenizers = {
            "prior_tokenizer": unsupervised_model.prior_model.tokenizer,
            "posterior_tokenizer": unsupervised_model.posterior_model.tokenizer,
            "decoder_tokenizer": unsupervised_model.decoder_model.tokenizer
        }

    set_seed(args)

    if (not args.eval_only):
        unsupervised_train_dataset = UnsupervisedDataset(
            args, tokenizers, split="train")
        unsupervised_eval_dataset = UnsupervisedDataset(
            args, tokenizers, split="val")

        Train(args, unsupervised_train_dataset,
              unsupervised_eval_dataset, unsupervised_model)
    else:
        unsupervised_eval_dataset = UnsupervisedDataset(
            args, tokenizers, labels_file=args.labels_file)
        results = Evaluate(args, unsupervised_eval_dataset, unsupervised_model)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))


if (__name__ == "__main__"):
    main()
