import json
import logging
import os
import sys
from functools import partial

import faiss
import numpy as np
import torch
from datasets import Features, Sequence, Value, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

from .data import pad_ids

logger = logging.getLogger(__name__)


class DatasetWalker:
    def __init__(self, args, split=None, labels_file=None, embed=True):
        if (labels_file == None):
            if (split == "train"):
                path = os.path.join(args.dataroot, "train.json")
            elif (split == "val"):
                path = os.path.join(args.dataroot, "val.json")
            elif (split == "test"):
                path = os.path.join(args.dataroot, "test.json")
        else:
            path = labels_file

        with open(path, "r") as f:
            self.dataset = json.load(f)

        if (embed):
            self.ctx_encoder = DPRContextEncoder.from_pretrained(
                args.document_encoder_model_name).cuda()
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                args.document_encoder_model_name)

            for i in tqdm(range(len(self.dataset))):
                self.dataset[i] = self.embed(
                    self.dataset[i], self.ctx_encoder, self.ctx_tokenizer)

    def embed(self, example, ctx_encoder, ctx_tokenizer):
        example["doc_embeddings"] = []
        titles = []
        texts = []
        for i in example["docs"]:
            titles.append(i["title"])
            texts.append(i["text"])
        input_ids = ctx_tokenizer(
            titles, texts, truncation=True, padding="longest", return_tensors="pt")["input_ids"]
        embeddings = ctx_encoder(
            input_ids.cuda(), return_dict=True).pooler_output

        example["doc_embeddings"] = embeddings.detach().cpu().numpy()

        return example

    def __iter__(self):
        for example in self.dataset:
            yield example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, data_walker=None):
        self.tokenizer = tokenizer

        self.cls = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("[CLS]"))[0]
        self.pad = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("[PAD]"))[0]
        self.sep = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("[SEP]"))[0]

        if (args.dialog):
            self.speaker1 = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize("<speaker1>"))[0]
            self.speaker2 = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize("<speaker2>"))[0]

        self.dataset_walker = data_walker
        self.dialog = args.dialog
        self.examples = self._create_examples()

    def _create_examples(self):
        logger.info("Creating examples")
        examples = []
        for i in tqdm(self.dataset_walker):
            y = i["response"]
            doc_id = i["doc_id"]
            qid = i["qid"]
            doc_embeddings = i["doc_embeddings"]

            if (self.dialog):
                x = i["dialog"]
                x_ids = None
                for j, t in enumerate(x):
                    t = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(t))
                    t = t + [self.sep]

                    if (x_ids == None):
                        x_ids = t
                    else:
                        x_ids += t
                x_ids = x_ids[:-1]
            else:
                x = i["query"]
                x_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(x))
            y_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(y))

            example = {
                "x_ids": x_ids,
                "y_ids": y_ids,
                "doc_embeddings": doc_embeddings,
                "doc_id": doc_id,
                "qid": qid
            }
            examples.append(example)

        return examples

    def __len__(self):
        return len(self.examples)


class PriorDataset(BaseDataset):
    def __init__(self, args, tokenizer, data_walker=None):
        super(PriorDataset, self).__init__(
            args, tokenizer, data_walker)

    def build_input_from_segments(self, example):
        input_ids = [self.cls] + example["x_ids"] + [self.sep]
        return input_ids

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = self.build_input_from_segments(example)

        d = {
            "example": example,
            "input_ids": input_ids,
            "doc_embeddings": example["doc_embeddings"]
        }
        return d


class PosteriorDataset(BaseDataset):
    def __init__(self, args, tokenizer, data_walker=None):
        super(PosteriorDataset, self).__init__(
            args, tokenizer, data_walker)

    def build_input_from_segments(self, example):
        input_ids = [self.cls] + example["x_ids"] + [self.sep]
        token_type_ids = len(input_ids) * [0]

        input_ids += example["y_ids"] + [self.sep]
        token_type_ids += (len(input_ids) - len(token_type_ids)) * [1]

        return input_ids, token_type_ids

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids, token_type_ids = self.build_input_from_segments(example)

        d = {
            "example": example,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "doc_embddings": example["doc_embeddings"]
        }
        return d


class DecoderDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, labels_file=None):
        self.tokenizer = tokenizer
        self.dataset_walker = DatasetWalker(
            args, split=split, labels_file=labels_file, embed=False)

        self.speaker1, self.speaker2 = self.tokenizer.convert_tokens_to_ids(
            ["<speaker1>", "<speaker2>"])

        self.dialog = args.dialog
        self.examples = self._create_examples()

    def _create_examples(self):
        logger.info("Creating examples")
        examples = []
        for i in tqdm(self.dataset_walker):
            y = i["response"]
            doc_id = i["doc_id"]
            qid = i["qid"]

            if (self.dialog):
                x = i["dialog"]
                x_ids = None
                for j, t in enumerate(x):
                    t = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(t))

                    if (j % 2 == 0):
                        t = [self.speaker1] + t
                    else:
                        t = [self.speaker2] + t

                    if (x_ids == None):
                        x_ids = t
                    else:
                        x_ids += t
            else:
                x = i["query"]
                x_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(x))
            y_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(y))

            example = {
                "x_ids": x_ids,
                "y_ids": y_ids,
                "doc_id": doc_id,
                "qid": qid,
                "docs": i["docs"]
            }
            examples.append(example)

        return examples

    def build_input_from_segments(self, example):
        input_ids = example["x_ids"]
        response_ids = example["y_ids"]
        return input_ids, response_ids

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids, response_ids = self.build_input_from_segments(example)

        d = {
            "example": example,
            "input_ids": input_ids,
            "response_ids": response_ids,
            "docs": example["docs"]
        }
        return d


class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizers, split=None, labels_file=None):
        self.prior_tokenizer = tokenizers["prior_tokenizer"]
        self.posterior_tokenizer = tokenizers["posterior_tokenizer"]
        self.decoder_tokenizer = tokenizers["decoder_tokenizer"]

        dataset_walker = DatasetWalker(
            args, split=split, labels_file=labels_file)

        self.prior_dataset = PriorDataset(
            args, self.prior_tokenizer, data_walker=dataset_walker)
        self.posterior_dataset = PosteriorDataset(
            args, self.posterior_tokenizer, data_walker=dataset_walker)
        self.decoder_dataset = DecoderDataset(
            args, self.decoder_tokenizer, split, labels_file=labels_file)

    def __getitem__(self, index):
        prior_example = self.prior_dataset[index]
        posterior_example = self.posterior_dataset[index]
        decoder_example = self.decoder_dataset[index]

        d = {
            "prior_input_ids": prior_example["input_ids"],
            "posterior_input_ids": posterior_example["input_ids"],
            "posterior_token_type_ids": posterior_example["token_type_ids"],
            "decoder_input_ids": decoder_example["input_ids"],
            "decoder_response_ids": decoder_example["response_ids"],
            "doc_id": prior_example["example"]["doc_id"],
            "qid": prior_example["example"]["qid"],
            "doc_embeddings": prior_example["doc_embeddings"],
            "docs": decoder_example["docs"]
        }
        return d

    def collate_fn(self, batch):
        prior_input_ids = [x["prior_input_ids"] for x in batch]
        prior_input_ids = torch.tensor(
            pad_ids(prior_input_ids, self.prior_dataset.pad))

        posterior_input_ids = [x["posterior_input_ids"] for x in batch]
        posterior_input_ids = torch.tensor(
            pad_ids(posterior_input_ids, self.posterior_dataset.pad))

        posterior_token_type_ids = [
            x["posterior_token_type_ids"] for x in batch]
        posterior_token_type_ids = torch.tensor(
            pad_ids(posterior_token_type_ids, self.posterior_dataset.pad))

        # Needs document so these ids are incomplete
        decoder_input_ids = [x["decoder_input_ids"] for x in batch]
        decoder_response_ids = [x["decoder_response_ids"] for x in batch]

        doc_ids = [x["doc_id"] for x in batch]
        q_ids = [x["qid"] for x in batch]

        doc_embeddings = [x["doc_embeddings"] for x in batch]
        doc_embeddings = torch.tensor(doc_embeddings)

        docs = [x["docs"] for x in batch]

        return prior_input_ids, posterior_input_ids, posterior_token_type_ids, decoder_input_ids, decoder_response_ids, doc_ids, q_ids, doc_embeddings, docs

    def __len__(self):
        return len(self.prior_dataset)
