import argparse
import copy
import json
import logging
import os
import random
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from .data import write_preds
from .dataset import (DecoderDataset, KnowledgeWalker, PosteriorDataset,
                      PriorDataset, UnsupervisedDataset)
from .models import DecoderModel, PosteriorModel, PriorModel, UnsupervisedModel
from .scorer import SelectionMetrics

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

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        local_steps = 0
        tr_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            global_step += 1
            model.train()

            loss, _, _, _, _, _ = model(batch)

            if (args.gradient_accumulation_steps > 1):
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if ((step + 1) % args.gradient_accumulation_steps == 0):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss/(local_steps+1))

        results = Evaluate(args, eval_dataset, model)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        # if (results["accuracy"] > best_acc):
        #     num_times_best_acc = 0
        #     best_model = copy.deepcopy(model)
        #     best_acc = results["accuracy"]
        # else:
        #     num_times_best_acc += 1
        #     if (num_times_best_acc == args.stopping_criteria):
        #         best_found = True
        #         best_model.save_model(args, "best")
        #         break

        model.save_model(args, "checkpoint-" + str(global_step))

    model.save_model(args, "")

    # if (not best_found):
    #     model.save_model(args, "best")


def Evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    selection_metrics = SelectionMetrics()

    d = {}
    with torch.no_grad():
        model.eval()

        for batch in epoch_iterator:
            prior_input_ids, _, _, decoder_input_ids, decoder_response_ids, doc_ids, q_ids = batch
            prior_dist, topk_documents_decoder_input_ids, topk_documents_ids = model.prior_model(
                prior_input_ids)

            for prior_dist_, topk_documents_ids_, doc_ids_, q_ids_ in zip(prior_dist, topk_documents_ids, doc_ids, q_ids):
                reciprocal_rank, recall_1, recall_k = selection_metrics.update(
                    topk_documents_ids_, doc_ids_)

                # NOTE if args.eval_only is true batch size should be 1
                if (args.eval_only):
                    # output_text = model.decoder_model.generate_from_1(
                    #     args, decoder_input_ids, topk_documents_decoder_input_ids)

                    d[q_ids_] = {
                        "prior_dist": prior_dist_.detach().cpu().numpy().tolist(),
                        "topk_document_ids": topk_documents_ids_,
                        # "generated_response_from_1": output_text,
                        # "selection_scores": {
                        #     "rr": reciprocal_rank,
                        #     "r@1": recall_1,
                        #     "r@" + str(selection_metrics.topk): recall_k
                        # }
                    }

    if (args.eval_only):
        write_preds(eval_dataset, args.output_file, d)

    results = selection_metrics.scores()

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
    parser.add_argument("--model_path", type=str,
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
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
            args.model_path, "prior", "params.json")
    logger.info("using params from " + args.params_file)

    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        args.update(params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints

    # Set seed
    set_seed(args)

    # Need decoder tokenizer for getting indexed passages
    decoder_model = DecoderModel(args)
    indexed_passages = KnowledgeWalker(args, decoder_model.tokenizer)
    del decoder_model

    unsupervised_model = UnsupervisedModel(
        args, indexed_passages.dataset).cuda()
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
        args.batch_size = 1
        unsupervised_eval_dataset = UnsupervisedDataset(
            args, tokenizers, labels_file=args.labels_file)
        results = Evaluate(args, unsupervised_eval_dataset, unsupervised_model)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))


if (__name__ == "__main__"):
    main()
