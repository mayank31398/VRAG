import argparse
import copy
import json
import logging
import os
import random
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchviz import make_dot
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from .data import write_preds
from .dataset import (DecoderDataset, KnowledgeWalker, PosteriorDataset,
                      PriorDataset, UnsupervisedDataset)
from .models import DecoderModel, PosteriorModel, PriorModel, UnsupervisedModel
from .scorer import Metrics

logger = logging.getLogger(__name__)


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

    optimizer = AdamW(model.GetParameters(),
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

            if (args.save_every != 0 and global_step % args.save_every == 0):
                model.save_model(args, "checkpoint-" + str(global_step))

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


# # NOTE evaluate posterior
# def Evaluate(args, eval_dataset, model):
#     eval_sampler = SequentialSampler(eval_dataset)
#     eval_dataloader = DataLoader(
#         eval_dataset,
#         sampler=eval_sampler,
#         batch_size=1,
#         collate_fn=eval_dataset.collate_fn
#     )

#     epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
#     metrics = Metrics()

#     d = {}
#     with torch.no_grad():
#         model.eval()

#         for batch in epoch_iterator:
#             _, posterior_input_ids, posterior_token_type_ids, decoder_input_ids, _, doc_ids, q_ids, has_cannot_answer = batch

#             posterior_logits, posterior_indices, _ = model.posterior_model(
#                 [posterior_input_ids.cuda(), posterior_token_type_ids.cuda()], args.topk)
#             posterior_dist = F.softmax(posterior_logits, dim=-1).cpu().tolist()[0]
#             posterior_indices = posterior_indices.cpu().tolist()

#             decoder_input_ids = decoder_input_ids[0]

#             if (args.n_gpus > 1):
#                 posterior_topk_documents_ids = model.posterior_model.module.indexed_passages.get_field_by_indices(
#                     posterior_indices, "id")[0]
#             else:
#                 posterior_topk_documents_ids = model.posterior_model.indexed_passages.get_field_by_indices(
#                     posterior_indices, "id")[0]

#             doc_ids = doc_ids[0]
#             q_ids = q_ids[0]

#             metrics.update_selection(posterior_topk_documents_ids, doc_ids)

#     results = metrics.scores()
#     return results


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
            prior_input_ids, _, _, decoder_input_ids, _, doc_ids, q_ids, has_cannot_answer = batch

            prior_logits, prior_indices, _ = model.prior_model(
                prior_input_ids.cuda(), args.topk)
            prior_dist = F.softmax(prior_logits, dim=-1).cpu().tolist()[0]
            prior_indices = prior_indices.cpu().tolist()

            decoder_input_ids = decoder_input_ids[0]

            if (args.n_gpus > 1):
                prior_topk_documents_ids = model.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_indices, "id")[0]

                prior_topk_documents_text = model.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_indices, "text")[0]
            else:
                prior_topk_documents_ids = model.prior_model.indexed_passages.get_field_by_indices(
                    prior_indices, "id")[0]

                prior_topk_documents_text = model.prior_model.indexed_passages.get_field_by_indices(
                    prior_indices, "text")[0]

            doc_ids = doc_ids[0]
            q_ids = q_ids[0]

            best_document_text = prior_topk_documents_text[0]

            metrics.update_selection(prior_topk_documents_ids, doc_ids)

            if (args.eval_only):
                if (args.n_gpus > 1):
                    output_text_from_1_doc = model.decoder_model.module.generate_from_1_doc(
                        args, decoder_input_ids, best_document_text)
                    output_text_from_k_docs = model.decoder_model.module.generate_from_k_docs(
                        args, decoder_input_ids, prior_topk_documents_text, prior_dist)
                else:
                    output_text_from_1_doc = model.decoder_model.generate_from_1_doc(
                        args, decoder_input_ids, best_document_text)
                    output_text_from_k_docs = model.decoder_model.generate_from_k_docs(
                        args, decoder_input_ids, prior_topk_documents_text, prior_dist)

                d[q_ids] = {
                    "prior_dist": prior_dist,
                    "topk_documents_ids": prior_topk_documents_ids,
                    "generated_response_from_1_doc": output_text_from_1_doc,
                    "generated_response_from_k_docs": output_text_from_k_docs
                }

    if (args.eval_only):
        write_preds(eval_dataset, args.output_file, d,
                    skip_cannot_answer=args.skip_cannot_answer)

    results = metrics.scores()
    return results


# # Calculate entropy
# def Evaluate(args, eval_dataset, model):
#     eval_sampler = SequentialSampler(eval_dataset)
#     eval_dataloader = DataLoader(
#         eval_dataset,
#         sampler=eval_sampler,
#         batch_size=1,
#         collate_fn=eval_dataset.collate_fn
#     )

#     epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
#     metrics = Metrics()

#     with torch.no_grad():
#         model.eval()

#         d = []
#         for doc in tqdm(model.prior_model.indexed_passages.dataset):
#             d.append(doc["embeddings"])
#         d = np.array(d)

#         p_z_given_x = []
#         p_z_given_xy = []
#         i = 0
#         for batch in epoch_iterator:
#             prior_input_ids, posterior_input_ids, posterior_token_type_ids, _, _, _, _, _ = batch

#             _, _, prior_question_embeddings = model.prior_model(
#                 prior_input_ids.cuda(), 1)
#             _, _, posterior_question_embeddings = model.posterior_model(
#                 [posterior_input_ids.cuda(), posterior_token_type_ids.cuda()], 1)

#             prior_question_embeddings = prior_question_embeddings.cpu().numpy()
#             posterior_question_embeddings = posterior_question_embeddings.cpu().numpy()
            
#             p_ = prior_question_embeddings @ d.T
#             p_ = torch.softmax(torch.tensor(p_), dim=-1).numpy()
#             p_z_given_x.append(p_)
            
#             p_ = posterior_question_embeddings @ d.T
#             p_ = torch.softmax(torch.tensor(p_), dim=-1).numpy()
#             p_z_given_xy.append(p_)
#         p_z_given_x = np.concatenate(p_z_given_x)
#         p_z_given_xy = np.concatenate(p_z_given_xy)

#         h_z_given_x = (-p_z_given_x * np.log(p_z_given_x)).sum(axis=1).mean()
#         h_z_given_xy = (-p_z_given_xy * np.log(p_z_given_xy)).sum(axis=1).mean()

#         print(h_z_given_x)
#         print(h_z_given_xy)
#         exit()

# # No document generation
# def Evaluate(args, eval_dataset, model):
#     eval_sampler = SequentialSampler(eval_dataset)
#     eval_dataloader = DataLoader(
#         eval_dataset,
#         sampler=eval_sampler,
#         batch_size=1,
#         collate_fn=eval_dataset.collate_fn
#     )

#     epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
#     metrics = Metrics()

#     d = {}
#     with torch.no_grad():
#         model.eval()

#         for batch in epoch_iterator:
#             _, _, _, decoder_input_ids, _, _, q_ids, _ = batch
#             decoder_input_ids = decoder_input_ids[0]

#             q_ids = q_ids[0]

#             if (args.eval_only):
#                 if (args.n_gpus > 1):
#                     output_text_from_1_doc = model.decoder_model.module.generate_from_1_doc(
#                         args, decoder_input_ids, "")
#                 else:
#                     output_text_from_1_doc = model.decoder_model.generate_from_1_doc(
#                         args, decoder_input_ids, "")

#                 d[q_ids] = {
#                     "prior_dist": [0.2] * 5,
#                     "topk_documents_ids": [0] * 5,
#                     "generated_response_from_1_doc": output_text_from_1_doc,
#                     "generated_response_from_k_docs": output_text_from_1_doc
#                 }

#     if (args.eval_only):
#         write_preds(eval_dataset, args.output_file, d,
#                     skip_cannot_answer=args.skip_cannot_answer)

#     exit()


def main():
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--model_path", type=str,
                        help="Name of the experiment, checkpoints will be stored here")
    parser.add_argument("--prior_path", type=str)
    parser.add_argument("--posterior_path", type=str)
    parser.add_argument("--decoder_path", type=str)
    parser.add_argument(
        "--build_index", action="store_true", help="Build index")
    parser.add_argument("--index_path", type=str, help="Path of the index")
    parser.add_argument("--n_gpus", type=int, default=1, help="Num GPUS")
    parser.add_argument("--dialog", action="store_true", help="dialog setting")
    parser.add_argument("--save_every", type=int,
                        help="save every nth step", default=0)
    parser.add_argument("--multitask", action="store_true",
                        help="Use multitask decoder")
    parser.add_argument("--weight", type=int,
                        help="weight for CANNOTANSWER", default=5)
    parser.add_argument("--weigh_cannot_answer",
                        action="store_true", help="use weight parameter")
    parser.add_argument("--skip_cannot_answer",
                        action="store_true", help="skip CANNOTANSWER")
    parser.add_argument("--fix_DPR", action="store_true",
                        help="fix DPR model weights")
    parser.add_argument("--fix_prior", action="store_true",
                        help="fix prior model weights")
    parser.add_argument("--fix_posterior", action="store_true",
                        help="fix posterior model weights")
    parser.add_argument("--fix_decoder", action="store_true",
                        help="fix DPR model weights")
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
