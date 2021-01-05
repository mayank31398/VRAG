import json
import logging
import os
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from transformers import (AutoConfig, AutoTokenizer, DPRQuestionEncoder,
                          GPT2LMHeadModel)

from .data import pad_ids
from .generate import top_filtering

logger = logging.getLogger(__name__)
EPSILON = 1e-10

def GetUnionKL(prior_model_outputs, posterior_model_outputs):
    prior_topk_documents_ids = prior_model_outputs["topk_documents_ids"]
    posterior_topk_documents_ids = posterior_model_outputs["topk_documents_ids"]
    prior_question_embeddings = prior_model_outputs["question_embeddings"]
    posterior_question_embeddings = posterior_model_outputs["question_embeddings"]
    prior_topk_documents_embeddings = prior_model_outputs["topk_documents_embeddings"]
    posterior_topk_documents_embeddings = posterior_model_outputs["topk_documents_embeddings"]

    batch_size = len(prior_topk_documents_ids)
    topk = len(prior_topk_documents_ids[0])

    KL = 0
    for i in range(batch_size):
        all_docs_embeds = []
        s = set()
        for j in range(topk):
            s.add(prior_topk_documents_ids[i][j])
            s.add(posterior_topk_documents_ids[i][j])
        for j in s:
            if (j in prior_topk_documents_ids[i]):
                k = prior_topk_documents_ids[i].index(j)
                all_docs_embeds.append(prior_topk_documents_embeddings[i][k])
            else:
                k = posterior_topk_documents_ids[i].index(j)
                all_docs_embeds.append(
                    posterior_topk_documents_embeddings[i][k])

        all_docs_embeds = torch.tensor(all_docs_embeds).T.cuda()

        prior_logits_full = prior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds
        posterior_logits_full = posterior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds

        prior_log_dist_full = F.log_softmax(
            prior_logits_full, dim=-1).squeeze()
        posterior_dist_full = F.softmax(
            posterior_logits_full, dim=-1).squeeze()

        KL += F.kl_div(prior_log_dist_full, posterior_dist_full)
    KL /= batch_size
    return KL


def GetPostKL(prior_model_outputs, posterior_model_outputs):
    prior_topk_documents_ids = prior_model_outputs["topk_documents_ids"]
    posterior_topk_documents_ids = posterior_model_outputs["topk_documents_ids"]
    prior_question_embeddings = prior_model_outputs["question_embeddings"]
    posterior_question_embeddings = posterior_model_outputs["question_embeddings"]
    prior_topk_documents_embeddings = prior_model_outputs["topk_documents_embeddings"]
    posterior_topk_documents_embeddings = posterior_model_outputs["topk_documents_embeddings"]

    batch_size = len(prior_topk_documents_ids)
    topk = len(prior_topk_documents_ids[0])

    KL = 0
    for i in range(batch_size):
        all_docs_embeds = posterior_topk_documents_embeddings[i]
        all_docs_embeds = torch.tensor(all_docs_embeds).T.cuda()

        prior_logits_full = prior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds
        posterior_logits_full = posterior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds

        prior_log_dist_full = F.log_softmax(
            prior_logits_full, dim=-1).squeeze()
        posterior_dist_full = F.softmax(
            posterior_logits_full, dim=-1).squeeze()

        KL += F.kl_div(prior_log_dist_full, posterior_dist_full)
    KL /= batch_size
    return KL


class PriorModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(PriorModel, self).__init__()
        self.max_length = 512

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "prior", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "prior", args.checkpoint))
            self.encoder = DPRQuestionEncoder.from_pretrained(os.path.join(
                args.model_path, "prior", args.checkpoint), config=self.config)
            logger.info("Loading prior model from %s", os.path.join(
                args.model_path, "prior", args.checkpoint))
        else:
            self.config = AutoConfig.from_pretrained(
                args.question_encoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.question_encoder_model_name)
            self.encoder = DPRQuestionEncoder.from_pretrained(
                args.question_encoder_model_name, config=self.config)
            logger.info("Loading prior model from %s",
                        args.question_encoder_model_name)

        self.indexed_passages = indexed_passages

    def retrieve(self, question_embeddings, topk):
        # does retrieval
        question_embeddings = question_embeddings.detach().cpu().numpy()
        retrievals = self.indexed_passages.get_nearest_examples_batch(
            'embeddings', question_embeddings, k=topk)
        return retrievals

    def forward(self, batch, topk):
        # batch_size x sequence_length
        input_ids = batch

        # batch_size x 768
        question_embeddings = self.encoder(
            input_ids=input_ids[:, :self.max_length]).pooler_output
        retrievals = self.retrieve(question_embeddings, topk)

        logits = []
        for i, document in enumerate(retrievals.total_examples):
            # (1 x 768) x (768 x topk)
            logits_ = question_embeddings[i].unsqueeze(
                0) @ torch.tensor(document["embeddings"]).T.cuda()
            logits.append(logits_)
        # batch_size x topk
        logits = torch.cat(logits)

        topk_documents_title = [i["title"] for i in retrievals.total_examples]
        topk_documents_text = [i["text"] for i in retrievals.total_examples]
        topk_documents_ids = [i["id"] for i in retrievals.total_examples]
        topk_documents_embeddings = [i["embeddings"]
                                     for i in retrievals.total_examples]

        return logits, topk_documents_title, topk_documents_text, topk_documents_ids, question_embeddings, topk_documents_embeddings

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "prior", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving prior model checkpoint to %s", output_dir)
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class PosteriorModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(PosteriorModel, self).__init__()
        self.max_length = 512

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "posterior", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "posterior", args.checkpoint))
            self.encoder = DPRQuestionEncoder.from_pretrained(os.path.join(
                args.model_path, "posterior", args.checkpoint), config=self.config)
            logger.info("Loading posterior model from %s", os.path.join(
                args.model_path, "posterior", args.checkpoint))
        else:
            self.config = AutoConfig.from_pretrained(
                args.question_encoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.question_encoder_model_name)
            self.encoder = DPRQuestionEncoder.from_pretrained(
                args.question_encoder_model_name, config=self.config)
            logger.info("Loading posterior model from %s",
                        args.question_encoder_model_name)

        self.indexed_passages = indexed_passages

    def retrieve(self, question_embeddings, topk):
        # does retrieval
        question_embeddings = question_embeddings.detach().cpu().numpy()
        retrievals = self.indexed_passages.get_nearest_examples_batch(
            'embeddings', question_embeddings, k=topk)
        return retrievals

    def forward(self, batch, topk):
        input_ids, token_type_ids = batch

        # batch_size x sequence_length
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()

        # batch_size x 768
        question_embeddings = self.encoder(
            input_ids=input_ids[:, :self.max_length], token_type_ids=token_type_ids[:, :self.max_length]).pooler_output
        retrievals = self.retrieve(question_embeddings, topk)

        logits = []
        for i, document in enumerate(retrievals.total_examples):
            # (1 x 768) x (768 x topk)
            logits_ = question_embeddings[i].unsqueeze(
                0) @ torch.tensor(document["embeddings"]).T.cuda()
            logits.append(logits_)
        # batch_size x topk
        logits = torch.cat(logits)

        topk_documents_title = [i["title"] for i in retrievals.total_examples]
        topk_documents_text = [i["text"] for i in retrievals.total_examples]
        topk_documents_ids = [i["id"] for i in retrievals.total_examples]
        topk_documents_embeddings = [i["embeddings"]
                                     for i in retrievals.total_examples]

        return logits, topk_documents_title, topk_documents_text, topk_documents_ids, question_embeddings, topk_documents_embeddings

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "posterior", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving posterior model checkpoint to %s", output_dir)
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class DecoderModel(nn.Module):
    def __init__(self, args):
        super(DecoderModel, self).__init__()

        self.SPECIAL_TOKENS = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<speaker1>", "<speaker2>"]
        }
        self.SPECIAL_TOKENS_VALUES = [
            "<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>"]
        self.max_length = 1024
        self.divide = args.divide

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "decoder", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "decoder", args.checkpoint))
            self.decoder = GPT2LMHeadModel.from_pretrained(os.path.join(
                args.model_path, "decoder", args.checkpoint), config=self.config)
            logger.info("Loading decoder model from %s", os.path.join(
                args.model_path, "decoder", args.checkpoint))
        else:
            self.config = AutoConfig.from_pretrained(args.decoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.decoder_model_name)
            self.decoder = GPT2LMHeadModel.from_pretrained(
                args.decoder_model_name, config=self.config)

            self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
            self.decoder.resize_token_embeddings(len(self.tokenizer))
            logger.info("Loading decoder model from %s",
                        args.decoder_model_name)

        self.bos = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2 = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"])

    def _dec(self, text):
        input_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text))
        return input_ids

    def _prepare_inputs(self, decoder_input_ids, decoder_response_ids, topk_documents_text, with_eos=True):
        batch_size = len(decoder_input_ids)
        topk = len(topk_documents_text[0])

        list_ids = []
        list_lms = []
        list_type_ids = []
        for i in range(batch_size):
            for j in range(topk):
                knowledge = self._dec(topk_documents_text[i][j])
                history = decoder_input_ids[i]
                response = decoder_response_ids[i]

                sequence = [[self.bos] + knowledge] + [history] + \
                    [response + ([self.eos] if with_eos else [])]
                sequence_with_speaker = [[self.speaker1 if (len(
                    sequence) - i) % 2 == 0 else self.speaker2] + s for i, s in enumerate(sequence[1:])]
                sequence = [sequence[0]] + sequence_with_speaker

                type_ids = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(
                    sequence) for _ in s]
                lms = ([-100] * sum(len(s)
                                    for s in sequence[:-1])) + [-100] + sequence[-1][1:]
                ids = list(chain(*sequence))

                list_ids.append(ids)
                list_lms.append(lms)
                list_type_ids.append(type_ids)

        list_ids = torch.tensor(pad_ids(list_ids, self.pad))
        list_lms = torch.tensor(pad_ids(list_lms, -100))
        list_type_ids = torch.tensor(pad_ids(list_type_ids, self.pad))

        list_ids = list_ids[:, :self.max_length]
        list_lms = list_lms[:, :self.max_length]
        list_type_ids = list_type_ids[:, :self.max_length]

        decoder_input_ids = list_ids.reshape(batch_size, topk, -1)
        decoder_response_ids = list_lms.reshape(batch_size, topk, -1)
        decoder_token_type_ids = list_type_ids.reshape(batch_size, topk, -1)

        return decoder_input_ids, decoder_response_ids, decoder_token_type_ids

    def compute_gen_loss_item(self, lm_logits, labels):
        batch_size, topk, sequence_length, _ = lm_logits.shape

        lm_logits = lm_logits.reshape(batch_size * topk, sequence_length, -1)
        labels = labels.reshape(batch_size * topk, -1)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

        loss = loss.reshape(batch_size * topk, -1)
        shift_labels = shift_labels.reshape(batch_size * topk, -1)

        loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
        loss = loss.reshape(batch_size, topk)

        return loss

    def forward(self, batch):
        decoder_input_ids, decoder_response_ids = batch

        decoder_input_ids = decoder_input_ids.cuda()
        decoder_response_ids = decoder_response_ids.cuda()

        batch_size, topk, sequence_length = decoder_input_ids.shape
        decoder_input_ids = decoder_input_ids.reshape(batch_size * topk, -1)

        decoder_model_outputs = self.decoder(
            input_ids=decoder_input_ids, token_type_ids=None)

        lm_logits = decoder_model_outputs[0]
        lm_logits = lm_logits.reshape(batch_size, topk, sequence_length, -1)
        # FIXME divide parameter
        lm_logits = lm_logits / self.divide

        loss = self.compute_gen_loss_item(lm_logits, decoder_response_ids)

        return loss, lm_logits

    # NOTE only works with batch size 1
    def generate_from_1_doc(self, args, decoder_input_ids, best_document_decoder_text):
        special_tokens_ids = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS_VALUES)
        current_output = []

        for i in range(args.generation_args["max_length"]):
            decoder_input_ids_, _, decoder_token_type_ids_ = self._prepare_inputs(
                [decoder_input_ids], [current_output], [[best_document_decoder_text]], with_eos=False)

            decoder_input_ids_ = decoder_input_ids_.cuda()
            decoder_token_type_ids_ = decoder_token_type_ids_.cuda()

            decoder_input_ids_ = decoder_input_ids_.reshape(1, -1)
            decoder_token_type_ids_ = decoder_token_type_ids_.reshape(1, -1)

            decoder_model_outputs = self.decoder(
                input_ids=decoder_input_ids_, token_type_ids=decoder_token_type_ids_)

            lm_logits = decoder_model_outputs[0]

            lm_logits = lm_logits[0, -1, :] / \
                args.generation_args["temperature"]
            lm_logits = top_filtering(
                lm_logits, top_k=args.generation_args["top_k"], top_p=args.generation_args["top_p"])
            probs = F.softmax(lm_logits, dim=-1)

            prev = torch.multinomial(probs, 1)
            if (i < args.generation_args["min_length"] and prev.item() in special_tokens_ids):
                while (prev.item() in special_tokens_ids):
                    if probs.max().item() == 1:
                        logger.warning(
                            "Warning: model generating special token with probability 1! Breaking...")
                        break
                    prev = torch.multinomial(probs, num_samples=1)

            if (prev.item() in special_tokens_ids):
                break
            current_output.append(prev.item())

        output_text = self.tokenizer.decode(
            current_output, skip_special_tokens=True)
        return output_text

    # NOTE only works with batch size 1
    def generate_from_k_docs(self, args, decoder_input_ids, topk_documents_decoder_text, prior_dist):
        special_tokens_ids = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS_VALUES)
        current_output = []

        for i in range(args.generation_args["max_length"]):
            decoder_input_ids_, _, decoder_token_type_ids_ = self._prepare_inputs(
                [decoder_input_ids], [current_output], [topk_documents_decoder_text], with_eos=False)

            decoder_input_ids_ = decoder_input_ids_.cuda()
            decoder_token_type_ids_ = decoder_token_type_ids_.cuda()

            decoder_input_ids_ = decoder_input_ids_.squeeze()
            decoder_token_type_ids_ = decoder_token_type_ids_.squeeze()

            decoder_model_outputs = self.decoder(
                input_ids=decoder_input_ids_, token_type_ids=decoder_token_type_ids_)

            lm_logits = decoder_model_outputs[0]

            probs = []
            for i in range(lm_logits.shape[0]):
                lm_logits_ = lm_logits[i, -1, :] / \
                    args.generation_args["temperature"]
                lm_logits_ = top_filtering(
                    lm_logits_, top_k=args.generation_args["top_k"], top_p=args.generation_args["top_p"])
                probs_ = F.softmax(lm_logits_, dim=-1)
                probs.append(probs_.unsqueeze(0))
            probs = torch.cat(probs, dim=0)
            probs = (torch.tensor(prior_dist).unsqueeze(0) @ probs.detach().cpu()).squeeze()

            prev = torch.multinomial(probs, 1)
            if (i < args.generation_args["min_length"] and prev.item() in special_tokens_ids):
                while (prev.item() in special_tokens_ids):
                    if probs.max().item() == 1:
                        logger.warning(
                            "Warning: model generating special token with probability 1! Breaking...")
                        break
                    prev = torch.multinomial(probs, num_samples=1)

            if (prev.item() in special_tokens_ids):
                break
            current_output.append(prev.item())

        output_text = self.tokenizer.decode(
            current_output, skip_special_tokens=True)
        return output_text

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "decoder", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving decoder model checkpoint to %s", output_dir)
        self.decoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class UnsupervisedModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(UnsupervisedModel, self).__init__()

        self.modeling_method = args.modeling_method
        if (self.modeling_method == "VRAG"):
            self.kl_beta = args.kl_beta
        self.topk = args.topk
        self.decoder_topk = args.decoder_topk
        self.prior_model = PriorModel(args, indexed_passages)
        self.posterior_model = PosteriorModel(args, indexed_passages)
        self.decoder_model = DecoderModel(args)

        if (args.n_gpus > 1):
            self.prior_model = nn.DataParallel(self.prior_model)
            self.posterior_model = nn.DataParallel(self.posterior_model)
            self.decoder_model = nn.DataParallel(self.decoder_model)

    def forward(self, batch):
        (prior_input_ids,
         posterior_input_ids,
         posterior_token_type_ids,
         decoder_input_ids,
         decoder_response_ids,
         doc_ids,
         q_ids) = batch

        if (self.modeling_method == "RAG"):
            prior_logits, _, prior_topk_documents_text, prior_topk_documents_ids, _, _ = self.prior_model(
                prior_input_ids.cuda(), self.topk)
            p_z_given_x = F.softmax(prior_logits, dim=-1) + EPSILON

            decoder_input_ids, decoder_response_ids, _ = self.decoder_model._prepare_inputs(
                decoder_input_ids, decoder_response_ids, prior_topk_documents_text)

            decoder_loss, _ = self.decoder_model(
                [decoder_input_ids.cuda(), decoder_response_ids.cuda()])

            p_y_given_zx = torch.exp(-decoder_loss) + EPSILON

            p_y_given_x = (p_z_given_x * p_y_given_zx).sum(dim=-1) + EPSILON
            loss = (-torch.log(p_y_given_x)).mean()

            print("loss =", loss)
            print("p(z|x) =", p_z_given_x)
            print("p(y|zx) =", p_y_given_zx)
            print("p(y|x) =", p_y_given_x)
            print("topk_documents_ids =", prior_topk_documents_ids)
            print("doc_ids =", doc_ids)
            print("q_ids =", q_ids)
            print()

            return loss
        elif (self.modeling_method == "VRAG"):
            posterior_logits, _, posterior_topk_documents_text, posterior_topk_documents_ids, posterior_question_embeddings, posterior_topk_documents_embeddings = self.posterior_model(
                [posterior_input_ids.cuda(), posterior_token_type_ids.cuda()], self.topk)

            _, _, _, prior_topk_documents_ids, prior_question_embeddings, prior_topk_documents_embeddings = self.prior_model(
                prior_input_ids.cuda(), self.topk)

            decoder_input_ids, decoder_response_ids, _ = self.decoder_model._prepare_inputs(
                decoder_input_ids, decoder_response_ids, posterior_topk_documents_text)

            decoder_loss, _ = self.decoder_model(
                [decoder_input_ids[:, :self.decoder_topk, :], decoder_response_ids[:, :self.decoder_topk, :]])

            posterior_dist = F.softmax(
                posterior_logits[:, :self.decoder_topk], dim=-1) + EPSILON

            e = (posterior_dist * decoder_loss).sum(dim=-1)
            loss = e.mean()

            prior_model_outputs = {
                "topk_documents_ids": prior_topk_documents_ids,
                "question_embeddings": prior_question_embeddings,
                "topk_documents_embeddings": prior_topk_documents_embeddings
            }
            posterior_model_outputs = {
                "topk_documents_ids": posterior_topk_documents_ids,
                "question_embeddings": posterior_question_embeddings,
                "topk_documents_embeddings": posterior_topk_documents_embeddings
            }

            KL = GetUnionKL(prior_model_outputs, posterior_model_outputs)
            loss += self.kl_beta * KL

            print("loss =", loss)
            print("KL =", KL)
            print("decoder_loss =", decoder_loss)
            print("posterior_dist =", posterior_dist)
            print("e =", e)
            print("topk_documents_ids =", posterior_topk_documents_ids)
            print("doc_ids =", doc_ids)
            print("q_ids =", q_ids)
            print()

            return loss

    def save_model(self, args, model_name):
        self.prior_model.save_model(args, model_name)
        self.posterior_model.save_model(args, model_name)
        self.decoder_model.save_model(args, model_name)

    def GetParameters(self):
        params = list(self.prior_model.parameters()) + \
            list(self.decoder_model.parameters())
        if (self.modeling_method == "VRAG"):
            params += list(self.posterior_model.parameters())
        return params
