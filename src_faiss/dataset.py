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


def embed(documents, ctx_encoder, ctx_tokenizer):
    input_ids = ctx_tokenizer(documents["title"], documents["text"],
                              truncation=True, padding="longest", return_tensors="pt")["input_ids"]
    embeddings = ctx_encoder(input_ids.cuda(), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


class KnowledgeWalker:
    def __init__(self, args):
        if (args.build_index):
            dataset = load_dataset("json", data_files=[
                args.knowledge_file], split="train")

            # And compute the embeddings
            ctx_encoder = DPRContextEncoder.from_pretrained(
                args.document_encoder_model_name).cuda()
            ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                args.document_encoder_model_name)

            new_features = Features(
                {
                    "title": Value("string"),
                    "text": Value("string"),
                    "id": Value("int32"),
                    "embeddings": Sequence(Value("float32"))
                }
            )  # optional, save as float32 instead of float64 to save space
            dataset = dataset.map(
                partial(embed, ctx_encoder=ctx_encoder,
                        ctx_tokenizer=ctx_tokenizer),
                batched=True,
                batch_size=args.processing_args["batch_size"],
                features=new_features
            )

            os.makedirs(os.path.join(args.index_path), exist_ok=True)
            dataset.save_to_disk(os.path.join(args.index_path, "dataset"))

            index = faiss.IndexHNSWFlat(
                args.index_args["dimensions"], args.index_args["links"], faiss.METRIC_INNER_PRODUCT)
            dataset.add_faiss_index("embeddings", custom_index=index)

            dataset.get_index("embeddings").save(
                os.path.join(args.index_path, "index.faiss"))

            exit()
        else:
            dataset = load_from_disk(os.path.join(args.index_path, "dataset"))
            dataset.load_faiss_index("embeddings", os.path.join(
                args.index_path, "index.faiss"))
            self.dataset = dataset
            self.inds = {}
            for i, e in enumerate(self.dataset):
                self.inds[e["text"]] = i

    def retrieve(self, question_embeddings, topk):
        question_embeddings = question_embeddings.detach().cpu().numpy()
        retrievals = self.dataset.get_nearest_examples_batch(
            'embeddings', question_embeddings, k=topk)

        topk_documents_text = [i["text"] for i in retrievals.total_examples]
        indices = []
        for i in range(len(topk_documents_text)):
            indices.append([])
            for j in range(len(topk_documents_text[0])):
                indices[i].append(self.inds[topk_documents_text[i][j]])

        return indices

    def get_field_by_indices(self, indices, field="text"):
        results = []
        for i in range(len(indices)):
            results.append([])
            for j in range(len(indices[0])):
                results[i].append(self.dataset[indices[i][j]][field])
        return results


class DatasetWalker:
    def __init__(self, args, split=None, labels_file=None):
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

    def __iter__(self):
        for example in self.dataset:
            yield example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split=None, labels_file=None):
        self.tokenizer = tokenizer
        self.dataset_walker = DatasetWalker(
            args, split=split, labels_file=labels_file)

        self.cls = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("[CLS]"))[0]
        self.pad = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("[PAD]"))[0]
        self.sep = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("[SEP]"))[0]

        self.examples = self._create_examples()

    def _create_examples(self):
        logger.info("Creating examples")
        examples = []
        for i in tqdm(self.dataset_walker):
            x = i["query"]
            y = i["response"]
            doc_id = i["doc_id"]
            qid = i["qid"]

            x_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(x))
            y_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(y))

            example = {
                "query": x,
                "response": y,
                "x_ids": x_ids,
                "y_ids": y_ids,
                "doc_id": doc_id,
                "qid": qid
            }
            examples.append(example)

        return examples

    def __len__(self):
        return len(self.examples)


class PriorDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, labels_file=None):
        super(PriorDataset, self).__init__(
            args, tokenizer, split, labels_file)

    def build_input_from_segments(self, example):
        input_ids = [self.cls] + example["x_ids"] + [self.sep]
        return input_ids

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = self.build_input_from_segments(example)

        d = {
            "example": example,
            "input_ids": input_ids
        }
        return d

    def collate_fn(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        input_ids = torch.tensor(pad_ids(input_ids, self.pad))

        # doc_ids = [x["example"]["doc_id"] for x in batch]
        # q_ids = [x["example"]["qid"] for x in batch]

        return input_ids


class PosteriorDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, labels_file=None):
        super(PosteriorDataset, self).__init__(
            args, tokenizer, split, labels_file)

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
            "token_type_ids": token_type_ids
        }
        return d

    def collate_fn(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        input_ids = torch.tensor(pad_ids(input_ids, self.pad))

        token_type_ids = [x["token_type_ids"] for x in batch]
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))

        # doc_ids = [x["example"]["doc_id"] for x in batch]
        # q_ids = [x["example"]["qid"] for x in batch]

        return input_ids, token_type_ids


class DecoderDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, labels_file=None):
        self.tokenizer = tokenizer
        self.dataset_walker = DatasetWalker(
            args, split=split, labels_file=labels_file)

        self.examples = self._create_examples()

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
            "response_ids": response_ids
        }
        return d

    def collate_fn(self, batch):
        # Needs document so these ids are incomplete
        input_ids = [x["input_ids"] for x in batch]
        response_ids = [x["response_ids"] for x in batch]

        # q_ids = [x["example"]["qid"] for x in batch]

        return input_ids, response_ids


class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizers, split=None, labels_file=None):
        self.prior_tokenizer = tokenizers["prior_tokenizer"]
        self.posterior_tokenizer = tokenizers["posterior_tokenizer"]
        self.decoder_tokenizer = tokenizers["decoder_tokenizer"]

        self.prior_dataset = PriorDataset(
            args, self.prior_tokenizer, split, labels_file=labels_file)
        self.posterior_dataset = PosteriorDataset(
            args, self.posterior_tokenizer, split, labels_file=labels_file)
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
            "qid": prior_example["example"]["qid"]
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

        return prior_input_ids, posterior_input_ids, posterior_token_type_ids, decoder_input_ids, decoder_response_ids, doc_ids, q_ids

    def __len__(self):
        return len(self.prior_dataset)
